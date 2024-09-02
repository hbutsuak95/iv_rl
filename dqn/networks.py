import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RiskNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, out_size, seed, fc1_units=64, fc2_units=64, model_type=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(RiskNetwork, self).__init__()
        self.model_type = model_type
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, out_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # if self.model_type:
        return F.softmax(self.fc3(x))

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, out_size, seed, fc1_units=64, fc2_units=64, use_risk=False, risk_size=None):
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
        self.use_risk = use_risk
        # print(state_size, risk_size, fc1_units)
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, out_size)
        if use_risk:
            self.fc_risk = nn.Linear(risk_size, 12)
            self.fc2 = nn.Linear(fc1_units+12, fc2_units)
        

    def forward(self, state, risk=None):
        """Build a network that maps state -> action values."""

        x = F.relu(self.fc1(state))
        if risk is not None:
            risk = F.relu(self.fc_risk(risk))
            x = torch.cat([x, risk], axis=-1)

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

