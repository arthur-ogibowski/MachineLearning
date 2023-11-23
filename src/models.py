from src import database

class LearningModel(db.Model):
    id = database.Column(database.Integer, primary_key=True)
    name = database.Column(database.String(100), nullable=False)
    parameters = database.Column(database.String(100), nullable=False)
    accuracy = database.Column(database.Float, nullable=False)
    precision = database.Column(database.Float, nullable=False)
    recall = database.Column(database.Float, nullable=False)
    f1 = database.Column(database.Float, nullable=False)
    date_created = database.Column(database.DateTime, default=database.func.current_timestamp())

    def __repr__(self):
        return f"MLModel('{self.name}', '{self.parameters}', '{self.date_created}')"