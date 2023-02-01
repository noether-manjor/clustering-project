def get_db_url(db):
    '''
    
    returns a formatted string containing username, password
    host and database for connecting to the mySQL server and the
    database indicated.
    
    
    requires 3 global strin variables:
    host(ip address, or website), password(login password), username (user login), 
    
    '''
    _user, _pass, _host = (username, password, host)
    
    return f"mysql+pymysql://{_user}:{_pass}@{_host}/{db}"

host = "157.230.209.171"
username = "noether_2034"
password = "nlysalRwRXaueFpl8MzE1DSTFdmcFcTV"
