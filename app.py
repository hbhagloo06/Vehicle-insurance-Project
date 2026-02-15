from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from typing import Optional
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.staticfiles import StaticFiles
from vehicle_insurance.constants import APP_HOST,PORT
from contextlib import asynccontextmanager
from vehicle_insurance.pipline.prediction_pipeline import VehicleData
from vehicle_insurance.entity.s3_estimator import Proj1Estimator
from vehicle_insurance.entity.config_entity import VehiclePredictorConfig
import asyncio
import pandas as pd

estimator=None

@asynccontextmanager
async def lifespan(app:FastAPI):
    global estimator
    prediction_pipeline_config = VehiclePredictorConfig()
    estimator = Proj1Estimator(
                bucket_name=prediction_pipeline_config.model_bucket_name,
                model_path=prediction_pipeline_config.model_file_key_path)
    await asyncio.to_thread(estimator.load_model)
    warmup_df = pd.DataFrame([{
        "Gender": "Male",
        "Age": 30,
        "Driving_License": 1,
        "Region_Code": 28,
        "Previously_Insured": 0,
        "Annual_Premium": 30000.0,
        "Policy_Sales_Channel": 152,
        "Vintage": 100,
        "Vehicle_Age": "1-2 Year",
        "Vehicle_Damage": "Yes",
    }])

    await asyncio.to_thread(estimator.predict, warmup_df)

    print("✅ Model loaded + warmed up")
    yield

app=FastAPI(lifespan=lifespan)


app.mount('/static',StaticFiles(directory='static'),name='static') #so when the user will enter the page the browser will send a request to fast api to load the css file
#and this will help it know where is the static file found.

templates=Jinja2Templates(directory='templates') #this will tell fast api how to handle and where to find the templates html when we request for them below

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],        # Allow requests from ANY website
    allow_credentials=True,     # Allow cookies/session IDs to be sent
    allow_methods=['*'],        # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=['*']         # Allow any custom headers in requests
)

class DataForm:
    def __init__(self, request: Request):
        self.request = request
        self.Gender: Optional[str] = None
        self.Age: Optional[int] = None
        self.Driving_License: Optional[int] = None
        self.Region_Code: Optional[int] = None
        self.Previously_Insured: Optional[int] = None
        self.Vehicle_Age: Optional[str] = None
        self.Vehicle_Damage: Optional[str] = None
        self.Annual_Premium: Optional[float] = None
        self.Policy_Sales_Channel: Optional[int] = None
        self.Vintage: Optional[int] = None

    async def get_vehicle_data(self):
        form = await self.request.form()
        self.Gender = form.get("Gender")
        self.Age = int(form.get("Age"))
        self.Driving_License = int(form.get("Driving_License"))
        self.Region_Code = int(form.get("Region_Code"))
        self.Previously_Insured = int(form.get("Previously_Insured"))
        self.Annual_Premium = float(form.get("Annual_Premium"))
        self.Policy_Sales_Channel = int(form.get("Policy_Sales_Channel"))
        self.Vintage = int(form.get("Vintage"))
        self.Vehicle_Age = form.get("Vehicle_Age")
        self.Vehicle_Damage = form.get("Vehicle_Damage")



@app.get('/',tags=['authentication'])
async def index(request:Request):

    return templates.TemplateResponse(name='vehicle_data.html',context={'request':request,'context':'Rendering'})

@app.post('/')
async def PredictRouteClient(request:Request):
    try:
        global estimator
        form=DataForm(request)
        await form.get_vehicle_data()

        vehicle_data = VehicleData(
                                    Gender= form.Gender,
                                    Age = form.Age,
                                    Driving_License = form.Driving_License,
                                    Region_Code = form.Region_Code,
                                    Previously_Insured = form.Previously_Insured,
                                    Annual_Premium = form.Annual_Premium,
                                    Policy_Sales_Channel = form.Policy_Sales_Channel,
                                    Vintage = form.Vintage,
                                    Vehicle_Age = form.Vehicle_Age,
                                    Vehicle_Damage = form.Vehicle_Damage
                                    )
        
        vehicle_data_df=vehicle_data.get_vehicle_input_data_frame()
        confidence_score,result=await asyncio.to_thread(estimator.predict_with_confidence_score,vehicle_data_df)

        is_yes = result[0] == 1
        status = "Likely to Respond" if is_yes else "Unlikely to Respond"

        return templates.TemplateResponse(
            "vehicle_data.html",
            {
                "request": request,
                "context": status,
                "confidence_score": confidence_score[0],
                "is_yes": is_yes
            },
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

# Main entry point to start the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app,host=APP_HOST,port=PORT)




