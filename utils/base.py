import yaml

def load_yaml(config_path):
    with open(config_path, "r") as file:
        try:
            yaml_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file