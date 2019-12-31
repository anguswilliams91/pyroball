import pyro
from pyro import distributions as dist
from pyro.infer import Trace_ELBO, SVI, Predictive
from pyro.optim import Adam
from pyro.poutine import condition
from pyro.contrib.autoguide import (
    AutoMultivariateNormal, AutoDiagonalNormal, AutoIAFNormal, AutoLaplaceApproximation
)

import torch

from pyroball.util import early_stopping


class SVIDynamicModel:
    def __init__(self):
        self.team_to_index = None
        self.index_to_team = None
        self.n_teams = None
        self.samples = None

    def model(self, home_team, away_team, gameweek):
        n_gameweeks = max(gameweek) + 1
        gamma = pyro.sample("gamma", dist.LogNormal(0, 1))
        mu_b = pyro.sample("mu_b", dist.Normal(0, 1))

        with pyro.plate("teams", self.n_teams):

            log_a0 = pyro.sample("log_a0", dist.Normal(0, 1))
            log_b0 = pyro.sample("log_b0", dist.Normal(mu_b, 1))
            sigma_rw = pyro.sample("sigma_rw", dist.HalfNormal(0.1))

            with pyro.plate("random_walk", n_gameweeks - 1):
                diffs_a = pyro.sample("diff_a", dist.Normal(0, sigma_rw))
                diffs_b = pyro.sample("diff_b", dist.Normal(0, sigma_rw))

            log_a0_t = log_a0 if log_a0.dim() == 2 else log_a0[None, :]
            diffs_a = torch.cat((log_a0_t, diffs_a), axis=0)
            log_a = torch.cumsum(diffs_a, axis=0)

            log_b0_t = log_b0 if log_b0.dim() == 2 else log_b0[None, :]
            diffs_b = torch.cat((log_b0_t, diffs_b), axis=0)
            log_b = torch.cumsum(diffs_b, axis=0)

        pyro.sample("log_a", dist.Delta(log_a), obs=log_a)
        pyro.sample("log_b", dist.Delta(log_b), obs=log_b)
        home_inds = torch.tensor([self.team_to_index[team] for team in home_team])
        away_inds = torch.tensor([self.team_to_index[team] for team in away_team])
        home_rate = torch.clamp(
            log_a[gameweek, home_inds] - log_b[gameweek, away_inds] + gamma, -7, 2
        )
        away_rate = torch.clamp(
            log_a[gameweek, away_inds] - log_b[gameweek, home_inds], -7, 2
        )

        pyro.sample("home_goals", dist.Poisson(torch.exp(home_rate)))
        pyro.sample("away_goals", dist.Poisson(torch.exp(away_rate)))

    def fit(
        self,
        df,
        max_iter=6000,
        patience=200,
        optimiser_settings={"lr": 1.0e-2},
        elbo_kwargs={"num_particles": 5},
    ):
        teams = sorted(list(set(df["home_team"]) | set(df["away_team"])))
        home_team = df["home_team"].values
        away_team = df["away_team"].values
        home_goals = torch.tensor(df["home_goals"].values, dtype=torch.float32)
        away_goals = torch.tensor(df["away_goals"].values, dtype=torch.float32)
        gameweek = ((df["date"] - df["date"].min()).dt.days // 7).values

        self.team_to_index = {team: i for i, team in enumerate(teams)}
        self.index_to_team = {value: key for key, value in self.team_to_index.items()}
        self.n_teams = len(teams)
        self.min_date = df["date"].min()

        conditioned_model = condition(
            self.model, data={"home_goals": home_goals, "away_goals": away_goals}
        )
        guide = AutoDiagonalNormal(conditioned_model)

        optimizer = Adam(optimiser_settings)
        elbo = Trace_ELBO(**elbo_kwargs)
        svi = SVI(conditioned_model, guide, optimizer, loss=elbo)

        pyro.clear_param_store()
        fitted_svi, losses = early_stopping(
            svi, home_team, away_team, gameweek, max_iter=max_iter, patience=patience
        )

        self.guide = guide

        return losses

    def _predict(self, home_team, away_team, dates, num_samples=100, seed=42):

        predictive = Predictive(
            self.model,
            guide=self.guide,
            num_samples=num_samples,
            return_sites=("home_goals", "away_goals"),
        )

        home_team = [home_team] if isinstance(home_team, str) else home_team
        away_team = [away_team] if isinstance(away_team, str) else away_team

        missing_teams = set(list(home_team) + list(away_team)) - set(self.team_to_index.keys())

        for team in missing_teams:
            new_index = max(self.team_to_index.values()) + 1
            self.team_to_index[team] = new_index
            self.index_to_team[new_index] = team
            self.n_teams += 1

        gameweek = (dates - self.min_date).dt.days // 7

        predictions = predictive.get_samples(home_team, away_team, gameweek)

        return (
            predictions["home_goals"].detach().numpy(),
            predictions["away_goals"].detach().numpy(),
        )
