"""Convenient development launcher for the learning assistant web app."""

import uvicorn


def main() -> None:
    """Start the ASGI server, similar to a Spring Boot application main method."""
    # Import the application by module path so reload and package imports work.
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
