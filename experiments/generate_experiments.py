import os

def load_template(experiments_name):
    template_path = experiments_name + ".yaml"
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found at {template_path}")
    with open(template_path, 'r') as file:
        template = file.read()
    return template

def main():
    for experiments_name in ["pod_dl_rom", "pod_nn", "dl_rom"]: 
        n = [1,3,6,9,16,24,32,64,96,128]
        base_exp = load_template(experiments_name)
        os.makedirs(experiments_name, exist_ok=True)
        for i in n:
            exp = base_exp.replace("[n]", str(i))
            exp = exp.replace("[v]", str(i))
            exp_path = os.path.join(experiments_name, f"exp_{i:03d}.yaml")
            with open(exp_path, 'w') as file:
                file.write(exp)
    
    

if __name__ == "__main__":
    main()