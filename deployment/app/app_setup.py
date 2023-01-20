from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi_scheduler import SchedulerAdmin
from fastapi_amis_admin.admin.settings import Settings
from fastapi_amis_admin.admin.site import AdminSite
import os
import shutil

# from main import router
# from app.main import router
import uvicorn

#############################
# fix for path, but very ugly
from pathlib import Path
import os
import sys

myDir = os.getcwd()
path = Path(f"{myDir}/app")
a = str(path.parent.absolute())
sys.path.append(a)

from deployment.app.main import router

##############################


def create_app():
    """
    Create the FastAPI by including router endpoints and mounting static files
    """
    app = FastAPI()
    app.include_router(router)
    app.mount(
        "/static", StaticFiles(directory="./deployment/app/static"), name="static"
    )

    site = AdminSite(
        settings=Settings(database_url_async="sqlite+aiosqlite:///amisadmin.db")
    )
    scheduler = SchedulerAdmin.bind(site)

    # @scheduler.scheduled_job('interval', seconds=60)
    @scheduler.scheduled_job("cron", hour=4, minute=0)
    def cron_task_test():
        """
        Every 4 hours the uploaded image and model files will be removed.
        """
        if os.path.exists("deployment/app/static/assets/images"):
            for content in os.listdir("deployment/app/static/assets/images"):
                if content not in [".DS_Store"]:
                    shutil.rmtree(f"deployment/app/static/assets/images/{content}")
        if os.path.exists("deployment/app/static/assets/models"):
            for content in os.listdir("deployment/app/static/assets/models"):
                if content not in ["default.pth", "default.ckpt", ".DS_Store"]:
                    shutil.rmtree(f"deployment/app/static/assets/models/{content}")

    @app.on_event("startup")
    async def startup():
        site.mount_app(app)
        # Start the scheduled task scheduler
        scheduler.start()

    return app


app = create_app()

if __name__ == "__main__":
    # Run application
    uvicorn.run(app, host="127.0.0.1", port=8000)