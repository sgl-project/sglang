import os
import sys
from typing import List

import av
from datasets import load_dataset


def find_video_files(video_dir) -> List[str]:
    if os.path.isfile(video_dir):
        return [video_dir]

    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, file))
            # if file is dir
            elif os.path.isdir(file):
                video_files.extend(find_video_files(file))
    return video_files


def video_frames(video_path, max_frames) -> int:
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    return min(total_frames, max_frames)


class Video:
    def __init__(self, video_path, num_frames):
        self.path = video_path
        self.num_frames = num_frames

    def __str__(self):
        return f"Video({self.path}, {self.num_frames})"

    def __iter__(self):
        return iter((self.path, self.num_frames))


class VideoPrompt(Video):
    def __init__(self, video_path, num_frames, prompt):
        super().__init__(video_path, num_frames)
        self.prompt = prompt

    def __str__(self):
        return f"VideoPrompt({self.path}, {self.num_frames}, {self.prompt})"

    def __iter__(self):
        return iter((self.path, self.num_frames, self.prompt))


class VideoLoader:
    pass


class VideoFileLoader(VideoLoader):
    """
    Load all the videos in a directory
    """

    def __init__(self, video_dir, batch_size=1, max_frames=sys.maxsize):
        super().__init__()
        self.video_dir = video_dir
        self.video_files = find_video_files(video_dir)
        self.batch_size = batch_size
        self.max_frames = max_frames
        print(f"batch_size: {batch_size}, max_frames: {max_frames}")

    def __iter__(self):  # (file, number of frames)
        if self.batch_size == 1:
            for video_file in self.video_files:
                yield Video(video_file, video_frames(video_file, self.max_frames))
        else:
            batch = []
            for video_file in self.video_files:
                video = Video(video_file, video_frames(video_file, self.max_frames))
                batch.append(video)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []


class NExTQALoader(VideoLoader):
    """
    Load vdideos and prompts from NExT dataset
    set: train, test or validation
    """

    def __init__(
        self, video_dir, batch_size=1, max_frames=sys.maxsize, dset="test", task="OE"
    ):
        """
        task: 'MV' or 'OE'
        """
        super().__init__()
        self.task = task
        print(f"Loading the {dset} data of {task} from lmms-lab/NExTQA")
        self.ds = load_dataset("lmms-lab/NExTQA", task)
        self.ds = self.ds[dset]

        # self.n = ds.num_rows
        self.video_dir = video_dir
        self.video_files = find_video_files(video_dir)
        self.video_to_path = dict()
        for video_file in self.video_files:
            video_id = video_file.split("/")[-1].split(".")[0]
            self.video_to_path[video_id] = video_file

        self.batch_size = batch_size
        self.max_frames = max_frames

    def get_video_prompt(self, entry, max_frames) -> VideoPrompt:
        # Get video
        video_id = entry["video"]
        video_path = self.video_to_path[video_id]
        assert os.path.exists(video_path), f"Video not found: {video_path}"
        num_frames = min(entry["frame_count"], max_frames)
        video = Video(video_path, num_frames)
        prompt = entry["question"] + "?"
        if self.task == "MC":  # add choices
            prompt += f' a0: {entry["a0"]}, a1: {entry["a1"]}, a2: {entry["a2"]}, a3: {entry["a3"]}'
        return VideoPrompt(video_path, num_frames, prompt)

    def __iter__(self):
        if self.batch_size == 1:
            for entry in self.ds:
                yield self.get_video_prompt(entry, self.max_frames)
        else:
            batch = []
            for entry in self.ds:
                video = self.get_video_prompt(entry, self.max_frames)
                batch.append(video)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []


# main
if __name__ == "__main__":
    video_dir = "./videos"
    # video_loader = VideoFileLoader(video_dir, batch_size=16)
    # for batch in video_loader:
    #     print(f"Number of videos in batch: {len(batch)}")
    #     for video_file, num_frames in batch:
    #         print(f"Video: {video_file} number of frames: {num_frames}")

    video_loader = NExTQALoader(video_dir, batch_size=16, dset="test", task="OE")
    for batch in video_loader:
        print(f"Number of videos in batch: {len(batch)}")
        for video_file, num_frames, prompt in batch:
            print(
                f"Video: {video_file} number of frames: {num_frames}, prompt: {prompt}"
            )
        # break
        # for video_file, prompt in batch:
        #     print(f"Video: {video_file} prompt: {prompt}")
        #     break
