def describe_class(description):
    """
    Decorator function to add a description to a class.

    This decorator adds a __description__ attribute to the decorated class,
    which can be used for documentation or introspection.

    Args:
        description: The description to add to the class.

    Returns:
        A decorator function that adds the description to the class.
    """

    def decorator(cls):
        cls.__description__ = description
        return cls

    return decorator
