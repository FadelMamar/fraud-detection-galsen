import time

def fit_model(model,X_train,y_train, return_execution_time=False):
    """
    Fit the model using the given configuration and data.
    """
    start_time=time.time()
    model.fit(X_train, y_train)
    training_execution_time=time.time()-start_time

    if return_execution_time:
        return model, training_execution_time
    
    return model
