import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class c51QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, n_atoms=101, v_min=-100, v_max=100):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = 4
        self.network = nn.Sequential(
            nn.Linear(state_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.n * n_atoms),
        )

    def forward(self, x, action=None):
        logits = self.network(x)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action.cpu().numpy()[0], pmfs[torch.arange(len(x)), action]



class TwoHeadQNetwork(QNetwork):
    """Actor (Policy) Model with 2 heads."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__(state_size, action_size, seed, fc1_units, fc2_units)
        self.fc4 = nn.Linear(fc2_units, action_size)

    def forward(self, state, is_train=False):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu, logvar = self.fc3(x), self.fc4(x)
        if is_train:
        	return mu, logvar 
        else:
        	return mu

def get_activation(name):
    activation_dict = {
        'relu': nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
        "logsoftmax": nn.LogSoftmax(dim=1),
    }
    return activation_dict[name]

class FearNet(nn.Module):
    def __init__(self, obs_size=64, fc1_size=64, fc2_size=64,\
                  fc3_size=64, fc4_size=64, out_size=2, batch_norm=True, activation='relu', model_type="state_risk", action_size=2):
        super().__init__()
        self.obs_size = obs_size
        self.batch_norm = batch_norm
        self.model_type = model_type
        self.fc1 = nn.Linear(obs_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.out = nn.Linear(fc2_size, out_size)

        ## Batch Norm layers
        self.bnorm1 = nn.BatchNorm1d(fc1_size)
        self.bnorm2 = nn.BatchNorm1d(fc2_size)

        # Activation functions
        self.activation = get_activation(activation)

        self.softmax = get_activation("softmax")
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, action=None):
        x = self.bnorm1(self.activation(self.fc1(x)))
        x = self.bnorm2(self.activation(self.fc2(x)))
        out = self.softmax(self.out(x))
        return out


class PriorNet(nn.Module):
    def __init__(self, state_size, action_size, seed,  fc1_units=64):
        super(PriorNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(state_size, action_size) #fc1_units, action_size)

    def forward(self, state):
        #x = F.relu(self.fc1(state))
        return self.fc2(state)


class QNet_with_prior(nn.Module):
    def __init__(self, state_size, action_size, seed, prior_scale=10, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        """
        super(QNet_with_prior, self).__init__()
        self.prior = PriorNet(state_size, action_size, seed, fc1_units) #+random.choice(list(range(42))))
        self.net = QNetwork(state_size, action_size, seed, fc1_units, fc2_units)
        self.prior_scale = prior_scale

    def forward(self, state):
        return self.net(state) + self.prior_scale*self.prior(state)



class MCDropQNet(QNetwork):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, p=0.5):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super().__init__(state_size, action_size, seed, fc1_units, fc2_units)
        self.p = p  # Dropout Probability

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.dropout(x, p=self.p, training=True, inplace=True)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.p, training=True, inplace=True)
        return self.fc3(x)

