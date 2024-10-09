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


class QCNNetwork(nn.Module):
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
        super(QCNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(kernel_size=3, padding="same", in_channels=1, out_channels=16)
        self.conv2 = nn.Conv2d(kernel_size=3, padding="same", in_channels=16, out_channels=16)
        self.fc1 = nn.Linear(8*6*16, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



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

