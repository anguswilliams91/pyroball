import logging

import pyro
from pyro import distributions as dist
from pyro.optim import Adam
from pyro.infer import Trace_ELBO, SVI, Predictive
from pyro.poutine import condition

import torch
import torch.distributions.constraints as constraints

logger = logging.getLogger(__name__)

class SVIDixonColesModel:

    def __init__(self):
        self.team_to_index = None
        self.index_to_team = None
        self.n_teams = None

    def model(self, home_team, away_team):

        sigma_a = pyro.sample("sigma_a", dist.HalfNormal(1.0))
        sigma_b = pyro.sample("sigma_b", dist.HalfNormal(1.0))
        mu_b = pyro.sample("mu_b", dist.Normal(0.0, 1.0))

        log_gamma = pyro.sample("log_gamma", dist.Normal(0, 1))

        log_a = pyro.sample("log_a", dist.Normal(torch.zeros(self.n_teams), sigma_a))
        log_b = pyro.sample("log_b", dist.Normal(torch.ones(self.n_teams) * mu_b, sigma_b))

        home_inds = torch.tensor([self.team_to_index[team] for team in home_team])
        away_inds = torch.tensor([self.team_to_index[team] for team in away_team])
        home_rate = torch.exp(log_a[home_inds] + log_b[away_inds] + log_gamma)
        away_rate = torch.exp(log_a[away_inds] + log_b[home_inds])
            
        with pyro.plate("matches", size=len(home_goals)) as ind:
            pyro.sample("home_goals", dist.Poisson(home_rate))
            pyro.sample("away_goals", dist.Poisson(away_rate))

    def guide(self, home_team, away_team):

        scale_sigma_a = pyro.param("scale_sigma_a", torch.tensor(0.1), constraint=constraints.positive)
        scale_sigma_b = pyro.param("scale_sigma_b", torch.tensor(0.1), constraint=constraints.positive)
        loc_sigma_a = pyro.param("loc_sigma_a", torch.tensor(0.1))
        loc_sigma_b = pyro.param("loc_sigma_b", torch.tensor(0.1))
        pyro.sample("sigma_a", dist.LogNormal(loc_sigma_a, scale_sigma_a))
        pyro.sample("sigma_b", dist.LogNormal(loc_sigma_b, scale_sigma_b))

        scale_mu_b = pyro.param("scale_mu_b", torch.tensor(1.0), constraint=constraints.positive)
        loc_mu_b = pyro.param("loc_mu_b", torch.tensor(0.0))  
        pyro.sample("mu_b", dist.Normal(loc_mu_b, scale_mu_b))     

        scale_gamma = pyro.param("scale_mu_b", torch.tensor(1.0), constraint=constraints.positive)
        loc_gamma = pyro.param("loc_mu_b", torch.tensor(0.0))  
        pyro.sample("log_gamma", dist.Normal(loc_gamma, scale_gamma))   

        loc_a = pyro.param("loc_a", torch.zeros(self.n_teams))
        loc_b = pyro.param("loc_b", torch.zeros(self.n_teams))
        scale_a = pyro.param("scale_a", torch.ones(self.n_teams), constraint=constraints.positive)
        scale_b = pyro.param("scale_b", torch.ones(self.n_teams), constraint=constraints.positive)
        
        pyro.sample("log_a", dist.Normal(loc_a, scale_a))
        pyro.sample("log_b", dist.Normal(loc_b, scale_b))

    def fit(
        self,
        df,
        n_steps=500,
        optimiser_settings={"lr": 1.0e-2}, 
        elbo_kwargs={"num_particles": 5}
    ):

        teams = sorted(list(set(df["home_team"]) | set(df["away_team"])))
        home_team = df["home_team"].values
        away_team = df["away_team"].values
        home_goals = torch.tensor(df["home_goals"].values, dtype=torch.float64)
        away_goals = torch.tensor(df["away_goals"].values, dtype=torch.float64)

        self.team_to_index = {team: i for i, team in enumerate(teams)}
        self.index_to_team =  {value: key for key, value in self.team_to_index.items()}
        self.n_teams = len(teams)

        conditioned_model = condition(self.model, data={"home_goals": home_goals, "away_goals": away_goals})

        optimizer = Adam(optimiser_settings)
        elbo = Trace_ELBO(**elbo_kwargs)
        svi = SVI(conditioned_model, self.guide, optimizer, loss=elbo)

        pyro.clear_param_store()
        losses = []
        for i in range(n_steps):
            loss = svi.step(home_goals, away_goals, home_team, away_team)
            losses.append(loss)
            if i % 50 == 0:
                logger.info(f"SVI loss: {loss}")

        return losses