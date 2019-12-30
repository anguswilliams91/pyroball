import numpyro as pyro
from numpyro import distributions as dist
import jax.numpy as np
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.handlers import condition
import jax.random as random


class HMCDynamicModel:
    def __init__(self):
        self.team_to_index = None
        self.index_to_team = None
        self.n_teams = None
        self.samples = None

    def model(self, home_team, away_team, gameweek):
        n_gameweeks = max(gameweek) + 1
        sigma_0 = pyro.sample("sigma_0", dist.HalfNormal(5))
        sigma_b = pyro.sample("sigma_b", dist.HalfNormal(5))
        gamma = pyro.sample("gamma", dist.LogNormal(0, 1))

        b = pyro.sample("b", dist.Normal(0, 1))

        loc_mu_b = pyro.sample("loc_mu_b", dist.Normal(0, 1))
        scale_mu_b = pyro.sample("scale_mu_b", dist.HalfNormal(1))

        with pyro.plate("teams", self.n_teams):

            log_a0 = pyro.sample("log_a0", dist.Normal(0, sigma_0))
            mu_b = pyro.sample(
                "mu_b",
                dist.TransformedDistribution(
                    dist.Normal(0, 1),
                    dist.transforms.AffineTransform(loc_mu_b, scale_mu_b),
                ),
            )
            sigma_rw = pyro.sample("sigma_rw", dist.HalfNormal(0.1))

            with pyro.plate("random_walk", n_gameweeks - 1):
                diffs = pyro.sample(
                    "diff",
                    dist.TransformedDistribution(
                        dist.Normal(0, 1), dist.transforms.AffineTransform(0, sigma_rw)
                    ),
                )

            diffs = np.vstack((log_a0, diffs))
            log_a = np.cumsum(diffs, axis=-2)

            with pyro.plate("weeks", n_gameweeks):
                log_b = pyro.sample(
                    "log_b",
                    dist.TransformedDistribution(
                        dist.Normal(0, 1),
                        dist.transforms.AffineTransform(mu_b + b * log_a, sigma_b),
                    ),
                )

        pyro.sample("log_a", dist.Delta(log_a), obs=log_a)
        home_inds = np.array([self.team_to_index[team] for team in home_team])
        away_inds = np.array([self.team_to_index[team] for team in away_team])
        home_rate = np.clip(
            log_a[gameweek, home_inds] - log_b[gameweek, away_inds] + gamma, -7, 2
        )
        away_rate = np.clip(
            log_a[gameweek, away_inds] - log_b[gameweek, home_inds], -7, 2
        )

        pyro.sample("home_goals", dist.Poisson(np.exp(home_rate)))
        pyro.sample("away_goals", dist.Poisson(np.exp(away_rate)))

    def fit(self, df, iter=500, seed=42):
        teams = sorted(list(set(df["home_team"]) | set(df["away_team"])))
        home_team = df["home_team"].values
        away_team = df["away_team"].values
        home_goals = df["home_goals"].values
        away_goals = df["away_goals"].values
        gameweek = ((df["date"] - df["date"].min()).dt.days // 7).values

        self.team_to_index = {team: i for i, team in enumerate(teams)}
        self.index_to_team = {value: key for key, value in self.team_to_index.items()}
        self.n_teams = len(teams)
        self.min_date = df["date"].min()

        conditioned_model = condition(
            self.model, param_map={"home_goals": home_goals, "away_goals": away_goals}
        )
        nuts_kernel = NUTS(conditioned_model)
        mcmc = MCMC(nuts_kernel, num_warmup=iter // 2, num_samples=iter)
        rng_key = random.PRNGKey(seed)
        mcmc.run(rng_key, home_team, away_team, gameweek)

        self.samples = mcmc.get_samples()
        mcmc.print_summary()
        return self

    def _predict(self, home_team, away_team, dates, num_samples=100, seed=42):

        predictive = Predictive(
            self.model,
            num_samples=num_samples,
            posterior_samples=self.samples,
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

        gameweek = (dates - self.min_date).dt.days // 7

        predictions = predictive.get_samples(
            random.PRNGKey(seed), home_team, away_team, gameweek
        )

        return predictions["home_goals"], predictions["away_goals"]
