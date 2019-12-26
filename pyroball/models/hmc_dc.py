import numpyro as pyro
from numpyro import distributions as dist
import jax.numpy as np
from numpyro.infer import NUTS, MCMC, Predictive
from numpyro.handlers import condition
import jax.random as random


class HMCDixonColesModel:
    def __init__(self):
        self.team_to_index = None
        self.index_to_team = None
        self.n_teams = None
        self.samples = None

    def model(self, home_team, away_team):

        sigma_a = pyro.sample("sigma_a", dist.HalfNormal(1.0))
        sigma_b = pyro.sample("sigma_b", dist.HalfNormal(1.0))
        mu_b = pyro.sample("mu_b", dist.Normal(0.0, 1.0))
        rho_raw = pyro.sample("rho_raw", dist.Beta(2, 2))
        rho = 2.0 * rho_raw - 1.0

        log_gamma = pyro.sample("log_gamma", dist.Normal(0, 1))

        with pyro.plate("teams", self.n_teams):
            abilities = pyro.sample(
                "abilities",
                dist.MultivariateNormal(
                    np.array([0.0, mu_b]),
                    covariance_matrix=np.array(
                        [
                            [sigma_a ** 2.0, rho * sigma_a * sigma_b],
                            [rho * sigma_a * sigma_b, sigma_b ** 2.0],
                        ]
                    ),
                )
            )
        
        log_a = abilities[:, 0]
        log_b = abilities[:, 1]
        home_inds = np.array([self.team_to_index[team] for team in home_team])
        away_inds = np.array([self.team_to_index[team] for team in away_team])
        home_rate = np.exp(log_a[home_inds] + log_b[away_inds] + log_gamma)
        away_rate = np.exp(log_a[away_inds] + log_b[home_inds])

        pyro.sample("home_goals", dist.Poisson(home_rate).to_event(1))
        pyro.sample("away_goals", dist.Poisson(away_rate).to_event(1))

    def fit(
        self,
        df,
        iter=500,
        seed=42
    ):
        teams = sorted(list(set(df["home_team"]) | set(df["away_team"])))
        home_team = df["home_team"].values
        away_team = df["away_team"].values
        home_goals = df["home_goals"].values
        away_goals = df["away_goals"].values

        self.team_to_index = {team: i for i, team in enumerate(teams)}
        self.index_to_team = {value: key for key, value in self.team_to_index.items()}
        self.n_teams = len(teams)

        conditioned_model = condition(
            self.model, param_map={"home_goals": home_goals, "away_goals": away_goals}
        )
        nuts_kernel = NUTS(conditioned_model)
        mcmc = MCMC(nuts_kernel, num_warmup=iter // 2, num_samples=iter)
        rng_key = random.PRNGKey(seed)
        mcmc.run(rng_key, home_team, away_team)
        
        self.samples = mcmc.get_samples()
        mcmc.print_summary()
        return self

    def _predict(self, home_team, away_team, num_samples=100, seed=42):

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

        predictions = predictive.get_samples(random.PRNGKey(seed), home_team, away_team)

        return predictions["home_goals"], predictions["away_goals"]