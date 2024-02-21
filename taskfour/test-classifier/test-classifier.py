import os
from dotenv import load_dotenv
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

def classify_images(prediction_endpoint, prediction_key, project_id, model_name, image_folder):
    """Classifies images using a Custom Vision model."""

    credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    prediction_client = CustomVisionPredictionClient(endpoint=prediction_endpoint, credentials=credentials)

    for image_path in os.listdir(image_folder):
        image_full_path = os.path.join(image_folder, image_path)
        try:
            with open(image_full_path, "rb") as image_file:
                image_data = image_file.read()
                results = prediction_client.classify_image(project_id, model_name, image_data)

            for prediction in results.predictions:
                if prediction.probability > 0.5:
                    print(f"{image_path}: {prediction.tag_name} ({prediction.probability:.0%})")

        except Exception as ex:
            print(f"Error processing image {image_path}: {ex}")

if __name__ == "__main__":
    load_dotenv()  # Load environment variables from a .env file

    try:
        prediction_endpoint = os.getenv("PredictionEndpoint")
        prediction_key = os.getenv("PredictionKey")
        project_id = os.getenv("ProjectID")
        model_name = os.getenv("ModelName")
        image_folder = r"taskfour\test-classifier\test-images"

        classify_images(prediction_endpoint, prediction_key, project_id, model_name, image_folder)

    except KeyError as e:
        print(f"Missing environment variable: {e}")
    except Exception as ex:
        print(f"An error occurred: {ex}")
