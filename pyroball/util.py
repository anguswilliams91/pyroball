import logging

logger = logging.getLogger(__name__)


def early_stopping(svi, *svi_args, max_iter=500, patience=20, **svi_kwargs):
    # stop early if SVI converged
    losses = []
    counter = 0 
    for i in range(max_iter):
        loss = svi.step(*svi_args, **svi_kwargs)
        losses.append(loss)
        if i > 0:
            if min(losses[:-1]) <= loss:
                counter += 1
                if counter == patience:
                    logger.info(f"Stopping early after {i} epochs.")
                    break
            else:
                counter = 0

        if i % 50 == 0:
            logger.info(f"Epoch {i} SVI loss: {loss}")
    
    return svi, losses
