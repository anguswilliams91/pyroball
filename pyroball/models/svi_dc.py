import logging

import pyro
from pyro import distributions as dist
from pyro.optim import Adam
from pyro.infer import Trace_ELBO, SVI, Predictive
from pyro.poutine import condition
from pyro.contrib.autoguide import AutoMultivariateNormal

import torch
import torch.distributions.constraints as constraints

from pyroball.util import early_stopping

logger = logging.getLogger(__name__)


class VariationalDixonColesModel:
    def __init__(self):
        self.team_to_index = None
        self.index_to_team = None
        self.n_teams = None
        self.guide = None

    def model(self, home_team, away_team):

        sigma_a = pyro.sample("sigma_a", dist.HalfNormal(1.0))
        sigma_b = pyro.sample("sigma_b", dist.HalfNormal(1.0))
        mu_b = pyro.sample("mu_b", dist.Normal(0.0, 1.0))
        rho_raw = pyro.sample("rho_raw", dist.Beta(2, 2))
        rho = pyro.deterministic("rho", 2.0 * rho_raw - 1.0)

        log_gamma = pyro.sample("log_gamma", dist.Normal(0, 1))

        with pyro.plate("teams", self.n_teams):
            abilities = pyro.sample(
                "abilities",
                dist.MultivariateNormal(
                    torch.tensor([0.0, mu_b]),
                    covariance_matrix=torch.tensor(
                        [
                            [sigma_a ** 2.0, rho * sigma_a * sigma_b],
                            [rho * sigma_a * sigma_b, sigma_b ** 2.0],
                        ]
                    ),
                ),
            )

        log_a = abilities[:, 0]
        log_b = abilities[:, 1]
        home_inds = torch.tensor([self.team_to_index[team] for team in home_team])
        away_inds = torch.tensor([self.team_to_index[team] for team in away_team])
        home_rate = torch.exp(log_a[home_inds] + log_b[away_inds] + log_gamma)
        away_rate = torch.exp(log_a[away_inds] + log_b[home_inds])

        pyro.sample("home_goals", dist.Poisson(home_rate))
        pyro.sample("away_goals", dist.Poisson(away_rate))

    def fit(
        self,
        df,
        max_iter=500,
        patience=5,
        optimiser_settings={"lr": 1.0e-2},
        elbo_kwargs={"num_particles": 5},
    ):

        teams = sorted(list(set(df["home_team"]) | set(df["away_team"])))
        home_team = df["home_team"].values
        away_team = df["away_team"].values
        home_goals = torch.tensor(df["home_goals"].values, dtype=torch.float64)
        away_goals = torch.tensor(df["away_goals"].values, dtype=torch.float64)

        self.team_to_index = {team: i for i, team in enumerate(teams)}
        self.index_to_team = {value: key for key, value in self.team_to_index.items()}
        self.n_teams = len(teams)

        conditioned_model = condition(
            self.model, data={"home_goals": home_goals, "away_goals": away_goals}
        )
        guide = AutoMultivariateNormal(conditioned_model)

        optimizer = Adam(optimiser_settings)
        elbo = Trace_ELBO(**elbo_kwargs)
        svi = SVI(conditioned_model, guide, optimizer, loss=elbo)

        pyro.clear_param_store()
        fitted_svi, losses = early_stopping(
            svi, home_team, away_team, max_iter=max_iter, patience=patience
        )

        self.guide = guide

        return losses

    def _predict(self, home_team, away_team, num_samples=100):

        predictive = Predictive(
            self.model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=("home_goals", "away_goals"),
        )

        home_team = [home_team] if isinstance(home_team, str) else home_team
        away_team = [away_team] if isinstance(away_team, str) else away_team

        missing_teams = set(home_team + away_team) - set(self.team_to_index.keys())

        for team in missing_teams:
            new_index = max(self.team_to_index.values()) + 1
            self.team_to_index[team] = new_index
            self.index_to_team[new_index] = team
            self.n_teams += 1

        simulations = predictive(home_team, away_team)

        home_goals = simulations["home_goals"].detach().numpy()
        away_goals = simulations["away_goals"].detach().numpy()

        return home_goals, away_goals
