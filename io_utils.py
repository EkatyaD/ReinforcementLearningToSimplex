def update_base_matrix(matrix_data):
    """
    Updates the base_matrix.py file with new matrix data
    """
    matrix_content = []
    matrix_content.append("import numpy as np\n\n")
    matrix_content.append("# Base matrix for simplex algorithm testing\n")
    matrix_content.append("# This matrix is generated automatically during training\n")
    matrix_content.append("BASE_MATRIX = np.array([\n")

    for i, row in enumerate(matrix_data):
        row_str = "    [" + ", ".join(f"{val:.3f}" for val in row) + "]"
        if i == len(matrix_data) - 1:
            matrix_content.append(f"{row_str}\n")
        else:
            matrix_content.append(f"{row_str},\n")

    matrix_content.append("])\n")

    with open('base_matrix.py', 'w') as f:
        f.writelines(matrix_content)


