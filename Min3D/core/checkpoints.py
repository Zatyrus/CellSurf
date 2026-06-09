## dependencies
import os
import lzma
import pickle
import datetime
from copy import deepcopy
from typing import Dict, Any, NoReturn, Union

## implementation of checkpoints base class
class Checkpoints:
    checkpoints: Dict[str, Dict[str, Any]]
    
    def __init__(self, **kwargs) -> NoReturn:
        self.checkpoints = {}

    # %% Checkpointing functions
    def create_checkpoint(self, slot: str = "latest") -> NoReturn:
        # create a DEEP COPY of the current object state
        self.checkpoints[slot] = {
            "checkpoint_time": datetime.datetime.now().strftime("%Y-%m-%dT_%H-%M-%S"),
            "data_set_state": deepcopy(self.dataset),
        }

    def write_checkpoint(self, out_path: str, slot: str = "latest") -> NoReturn:
        # check if path exists, else create
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        # create file name
        file_name = f"checkpoint_{slot}.xz"

        # join path and file name
        full_path = os.path.join(out_path, file_name)

        # if latest checkpoint does not exist, default to on_init
        if slot == "latest" and slot not in self.checkpoints:
            print("Latest checkpoint not found, defaulting to 'on_init' checkpoint.")
            slot = "on_init"

        # prepare data to be saved
        checkpoint = self.checkpoints[slot]

        # save using pickle with lzma compression
        with lzma.open(full_path, "wb") as f:
            pickle.dump(checkpoint, f)

    def write_snapshot(
        self, file_path: str, file_name: Union[str, None] = None
    ) -> NoReturn:
        # check if path exists, else create
        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        # create file name if none given
        if file_name is None:
            file_name = f"data_set_snapshot_{datetime.datetime.now().strftime('%Y-%m-%dT_%H-%M-%S')}.xz"

        # join path and file name
        full_path = os.path.join(file_path, file_name)

        # prepare data to be saved
        snapshot = {
            "snapshot_time": datetime.datetime.now().strftime("%Y-%m-%dT_%H-%M-%S"),
            "self.__dict__": self.__dict__,
        }

        # save using pickle with lzma compression
        with lzma.open(full_path, "wb") as f:
            pickle.dump(snapshot, f)

    def restore_checkpoint(self, slot: str = "latest") -> NoReturn:
        # catch error if latest checkpoint does not exist
        if slot not in self.checkpoints:
            print(f"Checkpoint '{slot}' not found. Cannot restore.")
            print("Available checkpoints:", list(self.checkpoints.keys()))
            return

        # restore the DEEP COPY of the checkpoint state
        self.dataset = deepcopy(self.checkpoints[slot]["data_set_state"])

    def restore_latest_checkpoint(self) -> NoReturn:
        return self.restore_checkpoint(slot="latest")

    def restore_original_data(self) -> NoReturn:
        return self.restore_checkpoint(slot="on_init")

    def reset_checkpoints(self) -> NoReturn:
        # open dialogue
        answer = input(
            "Are you sure you want to reset all checkpoints? This action cannot be undone. (y/n): "
        )
        if answer.lower() in ["y", "yes"]:
            self.checkpoints = {}
        else:
            print("Reset of checkpoints cancelled.")

    def load_checkpoint(self, checkpoint_path: str) -> NoReturn:
        # create checkpoint from current state to not lose it
        self.create_checkpoint(slot="latest")

        # load using pickle with lzma compression
        with lzma.open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        # restore the DEEP COPY of the checkpoint state
        self.dataset = deepcopy(checkpoint["data_set_state"])

    def restore_from_snapshot(self, snapshot_path: str) -> NoReturn:
        # load using pickle with lzma compression
        with lzma.open(snapshot_path, "rb") as f:
            snapshot = pickle.load(f)

        # restore all attributes from snapshot
        self.__dict__.update(snapshot["self.__dict__"])