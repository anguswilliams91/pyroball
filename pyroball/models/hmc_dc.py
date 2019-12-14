import numpyro
from numpyro import distributions as dist
import jax.numpy as np
from numpyro.infer import NUTS, MCMC
import jax.random as random


class HMCDixonColesModel:
    def __init__(self, teams):
        self.team_to_index = {team: i for i, team in enumerate(teams)}
        self.index_to_team = {value: key for key, value in self.team_to_index.items()}
        self.n_teams = len(teams)

    def model(self, home_goals, away_goals, home_team, away_team):

        sigma_a = numpyro.sample("sigma_a", dist.HalfNormal(1.0))
        sigma_b = numpyro.sample("sigma_b", dist.HalfNormal(1.0))
        mu_b = numpyro.sample("mu_b", dist.Normal(0.0, 1.0))

        log_gamma = numpyro.sample("log_gamma", dist.Normal(0, 1))

        log_a = numpyro.sample("log_a", dist.Normal(np.zeros(self.n_teams), sigma_a))
        log_b = numpyro.sample(
            "log_b", dist.Normal(np.ones(self.n_teams) * mu_b, sigma_b)
        )

        home_inds = np.array([self.team_to_index[team] for team in home_team])
        away_inds = np.array([self.team_to_index[team] for team in away_team])
        home_rate = np.exp(log_a[home_inds] + log_b[away_inds] + log_gamma)
        away_rate = np.exp(log_a[away_inds] + log_b[home_inds])

        with numpyro.plate("matches", size=len(home_goals)):
            numpyro.sample("home_goals", dist.Poisson(home_rate), obs=home_goals)
            numpyro.sample("away_goals", dist.Poisson(away_rate), obs=away_goals)

    def fit(
        self,
        n_steps=500,
        optimiser_settings={"lr": 1.0e-2},
        elbo_kwargs={"num_particles": 5},
    ):
        optimizer = Adam(optim_)
        elbo = Trace_ELBO(**elbo_kwargs)
        svi = SVI(self.model, self.guide, optimizer, loss=elbo)
