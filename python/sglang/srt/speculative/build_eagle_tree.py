import cutex
import torch

# parent_table [bs,topk*depth+)]
# selected_index [bs,draft_token_num-1)]
# verified_seq_len [bs]
# tree_mask [draft_token*(seq_len[0]+draft_token) | draft_token*(seq_len[1]+draft_token) | ..] = [sum(verified_seq_len)*draft_token+bs*draft_token*draft_token]
# positions [bs*draft_token]
# retrive_index [b, draft_token, depth+2]
kernels = cutex.SourceModule(
    """
//cuda
__global__ void build_tree(Tensor<long, 2> parent_list, Tensor<long, 2> selected_index, Tensor<int, 1> verified_seq_len,
        Tensor<bool, 1> tree_mask, Tensor<long, 1> positions, Tensor<long, 3> retrive_index, int topk, int depth, int draft_token_num) {
        int bid = blockIdx.x;
        int tid = threadIdx.x;
        if (tid >= draft_token_num){
            return;
        }
        int seq_tree_idx = draft_token_num * draft_token_num * bid;
        for(int i=0; i<bid; i++){
            seq_tree_idx += verified_seq_len[i] * draft_token_num;
        }
        int seq_len = verified_seq_len[bid];
        int token_tree_idx = seq_tree_idx + (seq_len+draft_token_num)*tid + seq_len + 1;
        for(int i=0; i<draft_token_num-1; i++){
            tree_mask[token_tree_idx+i] = false;
        }

        int position = 0;
        if (tid==0){
            positions[bid*draft_token_num] = seq_len;
            retrive_index[bid][0][0] = bid * draft_token_num;
            return;
        }

        int depends_order[10];

        int cur_position = tid-1;
        while(true){
            depends_order[position] = cur_position+1;
            position += 1;
            tree_mask[token_tree_idx+cur_position] = true;
            int parent_tb_idx = selected_index[bid][cur_position]/topk;
            if(parent_tb_idx==0){
                break;
            }

            int token_idx = parent_list[bid][parent_tb_idx];
            for(cur_position=0; cur_position<draft_token_num;cur_position++){
                if(selected_index[bid][cur_position]==token_idx){
                    break;
                }
            }
        }
        positions[bid*draft_token_num+tid] = position + seq_len;

        int is_leaf = 0;
        for(int i=1;i<draft_token_num;i++){
            if(tree_mask[seq_tree_idx + i * (draft_token_num+seq_len) + seq_len + tid])
            {
                is_leaf ++;
            }
        }
        if(is_leaf==1){
            for(int i=0; i<position; i++){
                retrive_index[bid][tid][position-i] = depends_order[i] + bid * draft_token_num;
            }
            retrive_index[bid][tid][0] = bid*draft_token_num;
        }



}
//!cuda
""",
    float_bits=16,  # change to 16 to use half precision as `float` type in the above source code.
    boundscheck=True,  # turning on for debug and off for performance (to use full threads of a block), default is on.
)


def build_tree_kernel(parent_list, top_score_index, seq_lens, topk, depth, draft_token):
    bs = seq_lens.numel()
    device = parent_list.device
    tree_mask = torch.full(
        (torch.sum(seq_lens).item() * draft_token + draft_token * draft_token * bs,),
        True,
        device=device,
    )
    retrive_index = torch.full(
        (bs, draft_token, depth + 2), -1, device=device, dtype=torch.long
    )
    positions = torch.empty((bs * draft_token,), device=device, dtype=torch.long)

    kernels.build_tree(
        parent_list,
        top_score_index,
        seq_lens.to(torch.int32),
        tree_mask,
        positions,
        retrive_index,
        topk,
        depth,
        draft_token,
        grid=(bs, 1, 1),
        block=(64, 1, 1),
    )
    index = retrive_index.sum(dim=-1) != -depth - 2
    cum_len = torch.cumsum(torch.sum(index, dim=-1), dim=-1)
    retrive_cum_len = torch.zeros(
        (cum_len.numel() + 1,), dtype=torch.int32, device="cuda"
    )
    retrive_cum_len[1:] = cum_len
    retrive_index = retrive_index[index]
    return tree_mask, positions, retrive_index, retrive_cum_len


if __name__ == "__main__":

    def findp(p_i, index, parent_list):
        pos = index // 10
        index_list = index.tolist()
        parent_list = parent_list.tolist()
        res = [p_i]
        while True:
            p = pos[p_i]
            if p == 0:
                break
            token_idx = parent_list[p]
            p_i = index_list.index(token_idx)
            res.append(p_i)
        return res

    def create_mask(seq_len, draft_token, index, parent_list, max_depth):
        mask = []
        positions = []
        retrive_index = []
        for i, lens in enumerate(seq_len.tolist()):
            first_mask = torch.full((lens + draft_token,), True)
            first_mask[-(draft_token - 1) :] = False
            positions.append(lens)
            mask.append(first_mask)
            seq_order = []
            first_index = torch.Tensor([0] + [-1] * (depth + 1)).cuda().to(torch.long)
            r_index = [first_index]
            for j in range(draft_token - 1):
                mask.append(torch.full((lens + 1,), True))
                idx = findp(j, index, parent_list)

                seq_order.append(idx)
                positions.append(len(idx) + seq_len)
                t = torch.full((draft_token - 1,), False)
                t[idx] = True
                mask.append(t)

            for i in range(1, draft_token - 1):
                is_leaf = 0
                for j in range(draft_token - 1):
                    if i in seq_order[j]:
                        is_leaf += 1

                if is_leaf == 1:
                    order_list = [0] + [x + 1 for x in seq_order[i][::-1]]
                    for _ in range(max_depth + 1 - len(seq_order[i])):
                        order_list.append(-1)
                    order = torch.Tensor(order_list).cuda().to(torch.long)
                    r_index.append(order)
            retrive_index.append(torch.stack(r_index))

        return (
            torch.cat(mask).cuda(),
            torch.Tensor(positions).cuda().to(torch.long),
            torch.stack(retrive_index),
        )

    index = (
        torch.Tensor(
            [
                0,
                1,
                2,
                3,
                10,
                11,
                12,
                13,
                20,
                21,
                22,
                30,
                110,
                130,
                150,
                160,
                210,
                211,
                212,
                213,
                214,
                215,
                216,
                217,
                218,
                219,
                220,
                230,
                310,
                311,
                312,
                313,
                314,
                315,
                316,
                317,
                320,
                321,
                322,
                330,
                360,
                380,
                390,
                410,
                411,
                412,
                413,
                414,
                415,
                416,
                417,
                418,
                419,
                420,
                421,
                422,
                423,
                430,
                431,
                440,
                441,
                460,
                470,
            ]
        )
        .to(torch.long)
        .cuda()
    )

    parent_list = (
        torch.Tensor(
            [
                -1,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                20,
                30,
                21,
                13,
                22,
                40,
                23,
                110,
                130,
                160,
                150,
                190,
                120,
                111,
                121,
                200,
                180,
                210,
                211,
                212,
                213,
                214,
                215,
                216,
                220,
                230,
                217,
                310,
                311,
                312,
                313,
                320,
                314,
                321,
                315,
                316,
                317,
            ]
        )
        .to(torch.long)
        .cuda()
    )

    verified_seq_len = torch.Tensor([47]).to(torch.long).cuda()
    bs = verified_seq_len.shape[0]
    topk = 10
    depth = 5  # depth <= 10
    draft_token = 64

    tree_mask = torch.full(
        (
            torch.sum(verified_seq_len).item() * draft_token
            + draft_token * draft_token * bs,
        ),
        True,
    ).cuda()
    retrive_index = torch.full(
        (bs, draft_token, depth + 2), -1, device="cuda", dtype=torch.long
    )
    positions = torch.empty((bs * draft_token,), device="cuda", dtype=torch.long)

    kernels.build_tree(
        parent_list.unsqueeze(0),
        index.unsqueeze(0),
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        topk,
        depth,
        draft_token,
        grid=(bs, 1, 1),
        block=(64, 1, 1),
    )
    retrive_index = retrive_index[retrive_index.sum(dim=-1) != -depth - 2]

    c_mask, c_positions, c_retive_index = create_mask(
        verified_seq_len, draft_token, index, parent_list, depth
    )

    assert torch.allclose(tree_mask, c_mask), "tree mask has error."
    assert torch.allclose(positions, c_positions), "positions has error."
    assert torch.allclose(retrive_index, c_retive_index), "retrive_index has error."
