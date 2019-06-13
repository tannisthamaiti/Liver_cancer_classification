import os
import pickle
from schema import Schema
from numpy import array
class IrisSVCModel(MLModel):
    '''A demonstration of how to use '''
    input_schema = Schema({'sepal_length': float,
                               'sepal_width': float,
                               'petal_length': float,
                               'petal_width': float}),
    
    "    # the output of the model will be one of three strings\n",
    output_schema = Schema({'species': Or("normal", "disease1", "disease2")})
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = open(os.path.join(dir_path, "model_files", "model_rf.pickle"), 'rb')
        self._rf_model = pickle.load(file)
        file.close()
    
    def predict(self, data):
    "        # calling the super method to validate against the input_schema\n",
        super().predict(data=data)
    
            # converting the incoming dictionary into a numpy array that can be accepted by the scikit-learn model\n",
        X = normalize(data)
    
            # making the prediction and extracting the result from the array\n",
        y_hat = int(self._rf_model.predict(X)[0])
    
            #converting the prediction into a string that will match the output schema of the model\n",
            # this list will map the output of the scikit-learn model to the output string expected by the schema\n",
        targets = ['normal', 'disease1', 'disease2']
        species = targets[y_hat]
    
        return {"species": species}
 