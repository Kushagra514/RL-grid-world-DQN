import torch 
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, n_actions):
            super().__init__()

            self.network = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),

                nn.ReLU(),

                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),

                nn.ReLU(),

                nn.Conv2d(
                    in_channels=64,
                    out_channels=128,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),

                nn.ReLU(),

                nn.Flatten(),

                nn.Linear(
                    128 * 13 * 13,
                    256
                ),

                nn.ReLU(),

                nn.Linear(
                    256,
                    n_actions
                )
            )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    model = DQN(n_actions = 4)

    dummy_input = torch.zeros(
        1,3,100,100
    )

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:",output.shape)
    print("Q-values",output)

