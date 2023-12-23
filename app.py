
from flask import Flask, request, jsonify



app = Flask(__name__)



@app.route('/predict', methods=['POST'])
def predict():
    from pyspark import SparkContext
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LinearSVC
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml import Pipeline
    from pyspark.ml import PipelineModel
    from pyspark.sql.functions import lit
    import os
    from pyspark.sql.functions import rand
    import findspark
    
    os.environ['HADOOP_HOME'] = r'C:\HADOOP'
    findspark.init()
    spark = SparkSession.builder.appName("BigDataProj").getOrCreate()
    
    #Data loading
    df = spark.read.csv(r'./train.csv',
                        header=True, inferSchema=True)
    
    df1 = spark.read.csv(r'./train.csv',
                     header=True, inferSchema=True)
    
    df = df.orderBy(rand())
    
    df1 = df1.orderBy(rand())
    
    # Tokenize the text
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    
    # Remove stop words
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered_words")
    
    # Define the number of batches
    num_batches = 10
    
    # Split the data into batches
    batch_size = df.count() // num_batches
    batches = [df.limit(batch_size).withColumn("batch_id", lit(i)) for i in range(num_batches)]

    # Initialize an AUC evaluator
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="is_depression")

    # Initialize an AUC accumulator
    total_auc = 0

    # Process and train the model on each batch
    for batch_df in batches:
        # Apply HashingTF to convert words to feature vectors
        hashingTF = HashingTF(inputCol=remover.getOutputCol(), outputCol="rawFeatures", numFeatures=1000)

        # Apply IDF to scale features
        idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")

        # Define the Logistic Regression model
        svm = LinearSVC(featuresCol="features", labelCol="is_depression")

        # Create a pipeline
        pipeline_svm = Pipeline(stages=[tokenizer, remover, hashingTF, idf, svm])

        # Fit the transformations and model on the batch
        model_svm = pipeline_svm.fit(batch_df)
        
        


    # Make predictions on the test data
    svm_predictions = model_svm.transform(df1)
    
    # Calculate AUC for this batch
    auc_svm = evaluator.evaluate(svm_predictions)


    # print("Average Area Under ROC (AUC): {:.4f}".format(average_auc))

    # Initialize variables to store TP, TN, FP, and FN
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # Calculate TP, TN, FP, FN
    tp += svm_predictions.filter((svm_predictions["is_depression"] == 1) & (svm_predictions["prediction"] == 1)).count()
    tn += svm_predictions.filter((svm_predictions["is_depression"] == 0) & (svm_predictions["prediction"] == 0)).count()
    fp += svm_predictions.filter((svm_predictions["is_depression"] == 0) & (svm_predictions["prediction"] == 1)).count()
    fn += svm_predictions.filter((svm_predictions["is_depression"] == 1) & (svm_predictions["prediction"] == 0)).count()

    # Calculate accuracy
    accuracy_svm = (tp + tn) / (tp + tn + fp + fn)

    # print("Accuracy: {:.2%}".format(accuracy_svm))
    
    
    
    data1 = request.get_json()
    text = data1['clean_text']
        
   
    # Create a DataFrame with the user input
    data = spark.createDataFrame([{"clean_text": text}])

    # Define the preprocessing steps used for training data
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="input_words")
    remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="input_filtered_words")

    # Create a pipeline for preprocessing
    preprocessing_pipeline = PipelineModel(stages=[tokenizer, remover])

    # Apply preprocessing to the user input
    preprocessed_df = preprocessing_pipeline.transform(data)

    # Add a dummy column to match the model's input schema
    preprocessed_df = preprocessed_df.withColumn("is_depression", lit(0))

    # Use the loaded model to make predictions on the preprocessed input
    predictions = model_svm.transform(preprocessed_df)

    # Extract the prediction result
    prediction = predictions.select("clean_text", "prediction").collect()[0]
    result = {"prediction":  int(prediction["prediction"])}

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


