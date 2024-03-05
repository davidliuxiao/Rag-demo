class PropertiesReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.properties = self.read_properties_file()

    def read_properties_file(self):
        properties = {}
        try:
            with open(self.file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    key, value = line.strip().split('|')
                    properties[key.strip()] = value.strip()
        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        return properties

    def get_property(self, key):
        return self.properties.get(key)


if __name__ == "__main__":
    file_path = 'example.properties'  # Specify the path to your properties file
    properties_reader = PropertiesReader(file_path)

    # Example usage of the class method
    key_to_lookup = 'key1'
    value = properties_reader.get_property(key_to_lookup)

    if value is not None:
        print(f"The value for key '{key_to_lookup}' is: {value}")
    else:
        print(f"Key '{key_to_lookup}' not found in the properties file.")