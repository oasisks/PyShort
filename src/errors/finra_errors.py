class FinraAuthError(Exception):
    """
    Exception raised for invalid authorization to Finra

    """
    def __init__(self, message: str = "Authorization Failed"):
        super().__init__(message)


class DateError(Exception):
    """
    Exception raised for invalid date being passed

    """
    def __init__(self, message: str = "Date is Invalid"):
        super().__init__(message)