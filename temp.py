from utils.knowledges import PropertiesReader

file_path = 'data/urls.properties'  # Specify the path to your properties file
properties_reader = PropertiesReader(file_path)

# Example usage of the class method
key_to_lookup = 'https://lde.tbe.taleo.net/lde01/ats/careers/requisition.jsp?org=BIS&cws=1&rid=1164'
value = properties_reader.get_property(key_to_lookup)

if value is not None:
    print(f"The value for key '{key_to_lookup}' is: {value}")
else:
    print(f"Key '{key_to_lookup}' not found in the properties file.")