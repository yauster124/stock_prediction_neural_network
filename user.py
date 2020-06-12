class User:
    """Created whenever a user logs in. Stores current user information."""
    def __init__(self, user_id, username):
        self._username = username
        self._user_id = user_id

    # Properties
    @property
    def username(self): return self._username
    @property
    def user_id(self): return self._user_id

