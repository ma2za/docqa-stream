# DocQA Stream

- [ ] Objective: Implement an API that can embed data into a vector store, accepts user queries, process them, and
  returns the results via HTTP streaming.

- [ ] UseCase: As a User, I want to upload my .pdf documents, so that I can ask questions and extract insights from it.

## Requirements:

- [ ] Use FastAPI for implementing the API service.
- [ ] You are free to choose the vector store of your preference (self-hosted or cloud based).
- [ ] HTTP Streaming: The response, based on the user's query, should be streamed back to the user.
- [ ] Containerization: Dockerize your service to ensure consistency across environments.
- [ ] Input: The primary input to this service will be a user query. Design an appropriate endpoint that accepts this
  query and initiates the processing.
- [ ] Insert Data: You can just create a simple script that manually inserts the data into the database. No need to
  create an endpoint for that.

## Notes

- [ ] You are free to use langchain, LamaIndex or OpenAI SDK.
- [ ] Bonus if you write tests (both for the endpoint part and for the output quality)
- [X] Host your code on a platform like GitHub or GitLab.