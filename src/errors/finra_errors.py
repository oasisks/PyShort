class FinraAuthInvalid(Exception):
    """
    Exception raised for invalid authorization to Finra

    """
    def __init__(self, message: str = "Authorization Failed"):
        super().__init__(message)
