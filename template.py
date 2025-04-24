import os
from pathlib import Path

package_name = "salary_prediction"

list_of_files = [
   ".github/workflows/ci.yaml",
   "src/__init__.py",
   f"src/{package_name}/__init__.py", 
   "src/components/__init__.py", 
   "src/components/data_ingestion.py", 
   "src/components/data_transformation.py", 
   "src/components/model_evalution.py", 
   "src/components/model_trainer.py", 
   "src/exception/__init__.py", 
    "src/exception/exception.py", 
      "src/logger/__init__.py", 
        "src/logger/logging.py", 
          "src/pipeline/__init__.py", 
            "src/pipeline/prediction_pipeline.py", 
            "src/pipeline/training_pipeline.py", 
"src/utils/__init__.py",
 "src/utils/utils.py", 
 "templates/index.html",
 "app.py",
   "requirements_dev.txt",
   "requirements.txt", 
   "setup.py",
   ".gitignore",
   "experiments/experiments.ipynb", 
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file

