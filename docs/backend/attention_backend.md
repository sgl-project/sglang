# Attention Backend

## Supporting matrix for different attention backend

| **Backend**      | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** | **Speed Rating**     |
|------------------|-------------------|-------------------|--------|--------------------|-----------------------|
| **FA3**          | ✅                | ✅                | ✅     | ✅                 | ⭐ ⭐ ⭐ ⭐             |
| **FlashInfer**   | ✅                | ✅                | ✅     | ✅                 | ⭐ ⭐ ⭐ ⭐             |
| **FlashMLA**     | ✅                | ❌                | ✅     | ❌                 | ⭐ ⭐ ⭐ ⭐ ⭐         |
| **Triton**       | ❌                | ✅                | ❌     | ❌                 | ⭐ ⭐ ⭐ ⭐          |
| **Torch Native** | ❌                | ❌                | ❌     | ❌                 | ⭐                    |

*Note: FlashMLA only supports page size = 64 case.*
