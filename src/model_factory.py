from surprise import ( 
  AlgoBase, 
  Prediction
)

from metrics import Metrics
from data_generator import DataGenerator


class Model: 
  def __init__(self, model: AlgoBase, name: str) -> None:
    self.model = model
    self.name = name
    self.is_fit = False


  def __str__(self) -> str:
    output = f"""
    Name Model:             { self.name } 
    Model:                  { self.model }
    Is the Model fit?       { self.is_fit }
    """
    return output
  


  def fit ( self, data: DataGenerator ) -> AlgoBase:
    
    trainset = data.get_trainset ( )
    fit_model = self.model.fit ( trainset )

    self.is_fit = True

    return fit_model


  def evaluate ( self, data: DataGenerator ) -> Metrics: 
    fit_model = self.fit ( data )
    
    testset = data.get_testset ( )
    predictions = fit_model.test ( testset )

    return Metrics ( predictions )
  


  def predict ( self, userID: int, itemID: int, r_ui: float = None, verbose: bool = False ) -> Prediction:
    
    return self.model.predict ( uid=userID, iid=itemID, r_ui=r_ui, verbose=False )


# sistema de recomendacion hibrido modelo paralelo metodo weighted
class HybridModel_Weighted ( AlgoBase ):

  def __init__ (self, models: list[ Model ], weights: list[ float ], **kwargs):

    super().__init__(**kwargs) 
    self.models = models
    self.weights = weights
  
  def fit (self, data_generator: DataGenerator) -> 'HybridModel_Weighted':

    trainset, _ = data_generator.get_train_test_set( )
    AlgoBase.fit ( self, trainset )
    
    for model in self.models:
      model.fit ( data_generator )
    
    return self
  
  def estimate ( self, user_id: int, item_id: int ) -> float:
    scores = 0
    for k in range ( len( self.models ) ):

      model_k = self.models[ k ]
      weight_k = self.weights[ k ]
      
      predition = model_k.predict ( userID=user_id, itemID=item_id )
      estimated_rating = predition.est
      
      scores += weight_k * estimated_rating

    return scores




# Esta clase funciona tanto para pobrar las metricas en diferentes modelos, como para hacer predicciones
class Factory:
  def __init__(self, data_generator: DataGenerator, models: list [ Model ]) -> None:
    self.dataset = data_generator
    self.models = models
  
  def add_model ( self, model: Model ) -> None:
    self.models.append ( model )
  
  
  def clean_models ( self ) -> None:
    self.models = [] 


  # TODO
  def evaluate ( self ) -> None:
    results = { }
    for model in self.models:
      print ( f'Evaluating { model.name }' )
      results [ model.name ] = model.evaluate( self.dataset )



def test ( ) -> None:
  pass

if __name__ == '__main__':
  test ( )
