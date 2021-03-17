# Build and Deploy a REST API Microservice with Python Flask and Docker on Heroku

# Objectives:

    - Build a simple but real-world useable REST API
    - Follow REST and Microservice Arch Best Practices
    - Deploy to a Docker Container

# Technologies used:

    - Python
    - Flask
    - Flask Restful
    - Docker

# Prerequisites:

    - A Foundational Understanding of Python
    - Acquaintance with the Flask Microframework
    - A Primary Understanding of Docker and Docker Container Management

# Build the Docker image locally and run to make sure that the service is running locally.

- Build Docker Image

  - docker build -t chatbot-rest-api:1.0 .

- Run Container / App
  - docker run -d -p 5000:5000 --name ChatBotRestAPI chatbot-rest-api:1.0

Opening the localhost on port 5000 should open our “not so complex” output. - http://localhost:5000

# Deploying the application to Heroku

Once you have your application running locally, we will start the actual work of deploying it in Heroku.
Login to Heroku container. It would open the browser and prompt you to login with your Heroku credentials if you are not logged in already. - heroku container:login

You would get a message as “Login Succeeded” if everything goes right.

It is time to create a new Heroku application. You could either choose a name or let Heroku create a magic name for your app. - heroku create
or - heroku create yourawesomeapp
You can skip above command, if would like to create application from Heruko GUI

Now it's time to execute application on Heroku - Make sure you are inside the folder where you created the 3 files app.py, requirements.txt and the almighty Dockerfile.

Execute the below command to create the container onto Heroku.

    - heroku container:push web --app conversational-chatbot-api

You will get the below messages as response which says that the Docker image has been built, tagged and successfully pushed.

We are almost there with completing our deployment. The container is pushed but not released yet. I’m not exactly sure what could be the reason to have it in pushed stage before releasing. Anyways, the below command would release the container.

    - heroku container:release web --app conversational-chatbot-api

Once it is released, you would get the message as done.

Now it is time to check out our awesome app running on Heroku.

- https://conversational-chatbot-api.herokuapp.com/

Note: This is a simple application but once we get all the things sorted, it could be expanded to any scale.
