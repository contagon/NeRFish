import numpy as np

class IterationState:
    def __init__(self, data_location) -> None:
        self.state = {"epoch": {}, "batch": {}}
        self.num_epochs = 0
        self.data_location = data_location

    def append(self, kind="batch", **kwargs):
        """Save data for later usage"""
        for k,v in kwargs.items():
            if k in self.state[kind]:
                self.state[kind][k].append(v)
            else:
                self.state[kind][k] = [v]

    def end_epoch(self, **kwargs):
        """Average over last batches of data, and save batch data as well"""
        # Average
        for k, v in self.state["batch"].items():
            avg = np.array(v).mean(axis=0)
            if k in self.state["epoch"]:
                self.state["epoch"][k].append(avg)
            else:
                self.state["epoch"][k] = [avg]

        self.state["batch"] = {}

        # Add in epoch data
        self.append(kind="epoch", **kwargs)
        self.num_epochs += 1

    def save(self, file):
        np.savez(file, dataset=self.data_location, **self.state["epoch"])