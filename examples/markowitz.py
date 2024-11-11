import cvxpy as cp
import numpy as np
from collections import namedtuple

MarkowitzProblem = namedtuple('MarkowitzProblem', ['problem', 'weights', 'cash', 'sqrtgamma_chol', 'gamma_kappa_trade', 'gamma_kappa_wold', 'ret'])
MarkowitzProblemBasic = namedtuple('MarkowitzProblemBasic', ['problem', 'weights', 'cash', 'sqrtgamma_chol', 'ret'])

def get_markowitz_problem(n_assets):

    weights = cp.Variable(n_assets)
    cash = cp.Variable()
    ret = cp.Parameter(n_assets, name='ret')
    sqrtgamma_chol = cp.Parameter((n_assets, n_assets), name='sqrtgamma_chol')
    gamma_kappa_trade = cp.Parameter(n_assets, nonneg=True, name='gamma_kappa_trade')
    gamma_kappa_wold = cp.Parameter(n_assets, name='gamma_kappa_wold')

    risk = cp.sum_squares(sqrtgamma_chol.T @ weights)
    turnover = cp.norm1(cp.multiply(gamma_kappa_trade, weights) - gamma_kappa_wold)

    objective = cp.Maximize(ret @ weights * 250 - risk * 250 - turnover * 250)
    constraints = [cp.sum(weights) + cash == 1, weights <= 0.1, weights >= -0.1, cp.norm1(weights) + cp.abs(cash) <= 1.6]
    # constraints = []

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    return MarkowitzProblem(problem=problem, weights=weights, cash=cash, sqrtgamma_chol=sqrtgamma_chol, gamma_kappa_trade=gamma_kappa_trade, gamma_kappa_wold=gamma_kappa_wold, ret=ret)

def get_markowitz_basic(n_assets):

    weights = cp.Variable(n_assets)
    cash = cp.Variable()
    ret = cp.Parameter(n_assets, name='ret')
    sqrtgamma_chol = cp.Parameter((n_assets, n_assets), name='sqrtgamma_chol')

    risk = cp.sum_squares(sqrtgamma_chol.T @ weights)

    objective = cp.Maximize(ret @ weights * 250 - risk * 250)
    constraints = [cp.sum(weights) + cash == 1]

    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    return MarkowitzProblemBasic(problem=problem, weights=weights, cash=cash, sqrtgamma_chol=sqrtgamma_chol, ret=ret)

BacktestBasic = namedtuple('BacktestBasic', ['weights', 'cash', 'gamma'])
def run_backtest_basic(means, choleskies, gamma):
    """
    Parameters
    ----------
    means : np.ndarray
        (n_samples, n_assets)
    choleskies : dict
        keys are the sample indices, values are the Cholesky factors of the
        covariance matrix
    omega : torch.Tensor of shape ()
        risk aversion parameter gamma_risk=exp(omega)
    
    Returns
    -------
    np.ndarray
        (n_samples, n_assets)
    """

    weights_t = np.ones(means.shape[1]) / means.shape[1]
    cash_t = np.array(0.0)

    _, n_assets = means.shape
    markowitz_problem = get_markowitz_basic(n_assets)
   
    weights = []
    cash = []

    from tqdm import tqdm
    for t in tqdm(means.index):
        mean_t = means.loc[t].values
        chol_t = choleskies[t].values
        sqrtgamma_chol_t = gamma ** 0.5 * chol_t

        markowitz_problem.ret.value = mean_t
        markowitz_problem.sqrtgamma_chol.value = sqrtgamma_chol_t

        markowitz_problem.problem.solve(solver='CLARABEL')
        
        weights_t, cash_t, = markowitz_problem.weights.value, markowitz_problem.cash.value
        weights.append(weights_t.reshape(1, -1))
        cash.append(cash_t)

    return BacktestBasic(weights=np.vstack(weights), cash=np.array(cash), gamma=gamma)

Backtest = namedtuple('Backtest', ['weights', 'cash', 'gamma_risk', 'gamma_trade'])
def run_backtest(means, spreads, choleskies, gamma_risk, gamma_trade):
    """
    Parameters
    ----------
    means : np.ndarray
        (n_samples, n_assets)
    choleskies : dict
        keys are the sample indices, values are the Cholesky factors of the
        covariance matrix
    omega : torch.Tensor of shape ()
        risk aversion parameter gamma_risk=exp(omega)
    
    Returns
    -------
    np.ndarray
        (n_samples, n_assets)
    """

    weights_t = np.ones(means.shape[1]) / means.shape[1]
    cash_t = np.array(0.0)

    _, n_assets = means.shape
    markowitz_problem = get_markowitz_problem(n_assets)

    weights = []
    cash = []
    from tqdm import tqdm
    for t in tqdm(means.index):
        mean_t = means.loc[t].values
        kappa_t = spreads.loc[t].values
        chol_t = choleskies[t].values
        gamma_kappa_trade = gamma_trade * kappa_t
        gamma_kappa_wold = gamma_trade * kappa_t * weights_t
        sqrtgamma_chol_t = gamma_risk ** 0.5 * chol_t

        markowitz_problem.ret.value = mean_t
        markowitz_problem.sqrtgamma_chol.value = sqrtgamma_chol_t
        markowitz_problem.gamma_kappa_trade.value = gamma_kappa_trade
        markowitz_problem.gamma_kappa_wold.value = gamma_kappa_wold

        markowitz_problem.problem.solve(solver='CLARABEL')

        weights_t, cash_t = markowitz_problem.weights.value, markowitz_problem.cash.value

        weights.append(weights_t.reshape(1, -1))
        cash.append(cash_t)

    return Backtest(weights=np.vstack(weights), cash=np.array(cash), gamma_risk=gamma_risk, gamma_trade=gamma_trade)


