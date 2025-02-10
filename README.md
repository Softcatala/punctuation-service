# punctuation-service

Softcatalà internal punctuation service


# How to run it locally

To build the docker container:

```shell
make docker-build
```

To run the docker container:

```shell
make docker-run
```

And open the URL: http://localhost:8000/check?text=%22Hola%20amics%20meus.%22
