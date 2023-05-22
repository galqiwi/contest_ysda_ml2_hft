import os


def read_file(filename: str) -> str:
    with open(filename, 'r') as file:
        return file.read()


def main():
    base = os.path.dirname(os.path.realpath(__file__))
    output = read_file(os.path.join(base, 'utils.py'))
    output += read_file(os.path.join(base, 'hft_model.py'))

    features = os.listdir(os.path.join(base, 'features'))
    features.sort(key=lambda feature_name: not feature_name.startswith('base_feature'))

    for feature in features:
        if not feature.endswith('.py'):
            continue
        output += read_file(os.path.join(base, 'features', feature))

    output += read_file(os.path.join(base, 'docker/solution.py'))

    output_lines = output.split('\n')

    def line_ok(line: str) -> bool:
        if line.startswith('from features'):
            return False
        if line.startswith('from utils'):
            return False
        if line.startswith('from hft_pipeline'):
            return False
        if line.startswith('import features'):
            return False
        if line.startswith('import utils'):
            return False
        if line.startswith('import hft_pipeline'):
            return False
        return True

    output = '\n'.join(filter(line_ok, output_lines))

    with open(os.path.join(base, 'solution.py'), 'w') as file:
        file.write(output)


if __name__ == '__main__':
    main()
