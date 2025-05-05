from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from pydantic import BaseModel, validator
from typing import Tuple, Optional
import numpy.typing as npt

class BostonDataConfig(BaseModel):
    test_size: float = 0.33
    random_state: Optional[int] = None

    @validator('test_size')
    def validate_test_size(cls, v):
        if not 0 < v < 1:
            raise ValueError('test_size must be between 0 and 1')
        return v

class BostonDataset(BaseModel):
    X: npt.NDArray
    y: npt.NDArray
    
    @validator('X')
    def validate_X(cls, v):
        if v.ndim != 2:
            raise ValueError('X must be a 2D array')
        if v.shape[1] != 13:  # Boston dataset has 13 features
            raise ValueError('X must have 13 features')
        return v
    
    @validator('y')
    def validate_y(cls, v):
        if v.ndim != 1:
            raise ValueError('y must be a 1D array')
        return v

def _preprocess_data(config: BostonDataConfig = BostonDataConfig()) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    X, y = datasets.load_boston(return_X_y=True)
    
    # Validate input data
    dataset = BostonDataset(X=X, y=y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, 
        dataset.y, 
        test_size=config.test_size,
        random_state=config.random_state
    )
    
    np.save('x_train.npy', X_train)
    np.save('x_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    print('Preprocessing data...')
    _preprocess_data()