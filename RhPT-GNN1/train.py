from GraphClassifGNN.Networks import *
from GraphClassifGNN import TrainValTest
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, LinearLR
from torch import device, load
import torch

from torch.optim import Adam
from torch_geometric import seed_everything
# print(torch_geometric.__version__)
import pickle

# ----------------------------------------------------------------------------------------------------------------------------------
# User-defined inputs: 
eval_period = 10  # This matters for the learning rate, as the exponential scheduler only updates every eval_period epochs

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
n_neurons = 24  # This is the size of the GNN hidden layer
n_output = 4  # Output dimension

record = False  # Chooses which Neptune page to send the data to

# case = 'case9'
# case = "case24_ieee_rts"
# case = "case24_ieee_rts_modified"
# case = "case9Q"
# case = "case9_modified"
# case = 'case9Q_test_N-1'
# case = 'case30'
# case = 'case30Q'
case = 'case118'
# case = 'case145'


points = 500  # Number of input points
batch_size = 20
gamma0 = 0.9996  # Learning rate decrease with exponential scheduler
lr0 = 0.0005  # Initial learning rate
epochs = 200000

# Penalty multipliers: mun0, muf0, muh0, betaf, betah
penalty_multiplers = [0.1, 0.001, 0.1, 1.00002,
                      1.00005]  # These multiplers are generally static except for mun0, which is a balancing term to decide of we want to focus more on cost or eq loss
training_params = [gamma0, lr0, case, epochs]  # Just for recording purposes
# ----------------------------------------------------------------------------------------------------------------------------------

# neptune = None
neptune = 1  # Uncomment this if you don't want the run to be sent to Neptune

if neptune == None:  #
    print('define the run')
    import neptune

    if record:
        run = neptune.init_run(
            # name = "Node_loss_test-1-AL",
            project="opfpinns/OpfPinns-MT-Experiments",
            api_token="YOUR_NEPTUNE_API_TOKEN_HERE",
        )
    else:
        run = neptune.init_run(  # Send the debugging stuff to a private workspace (Anna still has access)
            project="OpfPINNs2/OpfPINNS-MT",
            api_token="YOUR_NEPTUNE_API_TOKEN_HERE",
        )
else:
    print('dont define the run')
    run = 0

names = ["train", "val", "test"]

if run != 0:
    run["parameters/gamma0"] = gamma0
    run["parameters/lr0"] = lr0

if __name__ == "__main__":
    print('Loading {}_{}_{}.pt'.format(case, points, batch_size))

    train_loader = torch.load("Input_Data/train_loader_{}_{}_{}.pt".format(case, points, batch_size))
    val_loader = torch.load("Input_Data/val_loader_{}_{}_{}.pt".format(case, points, batch_size))
    test_loader = torch.load("Input_Data/test_loader_{}_{}_{}.pt".format(case, points, batch_size))
    with open('Input_Data/characteristics_{}.pkl'.format(case), 'rb') as f:
        characteristics = pickle.load(f)

    # Loads the data
    datasets = [
        train_loader,
        val_loader,
        test_loader,
    ]

    # seed_everything(2000)
    # Initialize model with characteristics for hard constraints (Section 2.4)
    model = NetGAT(
        train_loader.dataset[0].num_features,
        n_neurons,
        n_output,
        characteristics=characteristics
    ).to(device)

    # If you want to load a trained model for transfer learning:

    # model = torch.load("Results\Saved_Models\{}_2GAT\model_{}_2GAT.pt".format(case,case))

    optimizer = Adam(model.parameters(), lr=lr0, weight_decay=0.0001)
    scheduler = ExponentialLR(optimizer, verbose=False, gamma=gamma0)

    loaders = datasets

    TrainValTest.train_regr_pinn_correct(
        loaders,
        characteristics,
        model,
        optimizer,
        device=device,
        eval_period=eval_period,
        run=run,
        scheduler=scheduler,
        training_params=training_params,
        batch_size=batch_size,
        augmented_epochs=epochs,
        penalty_multipliers=penalty_multiplers
    )
