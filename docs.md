
my_pipeline/
│
├── api/
│   ├── __init__.py
│   ├── app.py             # Main API application
│   └── routes.py          # API route definitions
│
├── models/
│   ├── __init__.py
│   ├── models.py          # Contains Stable Diffusion, CLIP, and SAM classes
│   ├── model_pipeline.py   # Contains the Pipeline class that integrates the models
│   └── utils.py            # contains all the utility functions
│   
├── requirements.txt         # Required Python packages
├── config.py                # Configuration settings (e.g., model paths, device settings)
├── main.py                  # Entry point for running the pipeline (optional)
└── README.md                # Project documentation

main the entry point for running the application or running the pipeline independently from the API.
app.py is the point from where the api will run