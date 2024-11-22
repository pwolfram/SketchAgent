import xml.etree.ElementTree as ET
import re
import ast
import numpy as np
from IPython.display import SVG, display
from PIL import Image, ImageDraw, ImageFont
import math
from io import BytesIO
import base64


def create_grid_image(res=50, cell_size=12, header_size=12):
    # Define the size of the grid
    rows = res
    cols = res
    
    img_width = (cols + 1) * cell_size
    img_height = (rows + 1) * cell_size
    
    # Create a new image with a white background
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", header_size*0.85)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw the headers
    for j in range(cols):
        # Draw column header (letters)
        text = str(j +1)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (j + 1) * cell_size + (cell_size - text_width) / 2
        text_y = img_height - cell_size # - (cell_size - text_height) / 2
        draw.text((text_x, text_y), text, fill="black", font=font)
    
    for i in range(rows):
        # Draw row header (numbers)
        text = str(rows - i)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (cell_size - text_width) / 2
        text_y = i * cell_size + (cell_size - text_height) / 2 - 0.2*text_height
        draw.text((text_x, text_y), text, fill="black", font=font)
    
    # Draw the grid
    i = 1
    draw.line([(i * cell_size, 0), (i * cell_size, img_height)], fill="black")
    # Horizontal lines
    draw.line([(0, img_height -  cell_size), (img_width, img_height -  cell_size)], fill="black")
    
    positions={}
    # Draw the grid
    for i in range(rows)[::-1]:
        for j in range(cols):
            # Draw cell border
            if j == 0:
                draw.rectangle([(j + 0) * cell_size, (i + 0) * cell_size, (j + 1) * cell_size, (i + 1) * cell_size], outline="black")
            if i == rows - 1:
                draw.rectangle([(j + 0) * cell_size, (i + 1) * cell_size, (j + 1) * cell_size, (i + 2) * cell_size], outline="black")
    
            # Calculate the position of the text
            text = f"x{j + 1}y{i + 1}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (j + 1) * cell_size + (cell_size - text_width) / 2
            text_y = (i + 1) * cell_size + (cell_size - text_height) / 2
            
            point_radius = 5
            center_y = int(img_height - cell_size - (i * cell_size) - cell_size / 2)
            center_x = int(j * cell_size + cell_size / 2 + cell_size)
            point_radius = 2
            positions[text] = (center_x, center_y)

    return img, positions

def image_to_str(image: Image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    image = base64.b64encode(buffer.read()).decode('utf-8')
    return image


def cells_to_pixels(res=50, cell_size=12, header_size=12):
    # Define the size of the grid
    rows = res
    cols = res
    
    img_width = (cols + 1) * cell_size
    img_height = (rows + 1) * cell_size

    positions={}
    # Draw the grid
    for i in range(rows)[::-1]:
        for j in range(cols):
            # Calculate the position of the text
            text = f"x{j + 1}y{i + 1}"
            
            point_radius = 5
            center_y = int(img_height - cell_size - (i * cell_size) - cell_size / 2)
            center_x = int(j * cell_size + cell_size / 2 + cell_size)
            point_radius = 2
            positions[text] = (center_x, center_y)

    return positions


def extract_svg_paths_with_control_points(svg_content):
    """
    Given an SVG code (containing svg tags, group tags, and paths), extracts a list of control points for each group.
    Rturns a list of [{'group_id': 'id', 'paths':  [array([[x,y], [x,y]..]]), array([[x,y], [x,y]..]])]}]
    """
    # Parse the SVG content
    root = ET.fromstring(svg_content)
    
    # A list to store the extracted control points and group ids
    groups_with_paths = []
    
    # Iterate over all group elements in the SVG
    for group in root.findall(".//{http://www.w3.org/2000/svg}g"):
        group_id = group.get("id")
        paths_in_group = []
        
        # Iterate over all path elements inside this group
        for path in group.findall("{http://www.w3.org/2000/svg}path"):
            d_attr = path.get("d")
            
            # Match the Move command (M), Cubic Bezier Curve (C), Quadratic Bezier (Q), and Line (L) commands
            # 'M' Move to command (start point)
            move_match = re.search(r'M\s*([-\d.]+)[,\s]+([-\d.]+)', d_attr)
            if move_match:
                start_x, start_y = map(float, move_match.groups())
            
            # 'C' Cubic Bezier Curve command (control points and end point)
            curve_matches = re.findall(r'C\s*([-\d.]+)[,\s]+([-\d.]+)[,\s]+([-\d.]+)[,\s]+([-\d.]+)[,\s]+([-\d.]+)[,\s]+([-\d.]+)', d_attr)
            for match in curve_matches:
                x1, y1, x2, y2, x, y = map(float, match)
                points = [[start_x, start_y], [x1, y1], [x2, y2], [x, y]]
            
            # 'Q' Quadratic Bezier Curve command (control point and end point)
            quadratic_matches = re.findall(r'Q\s*([-\d.]+)[,\s]+([-\d.]+)[,\s]+([-\d.]+)[,\s]+([-\d.]+)', d_attr)
            for match in quadratic_matches:
                x1, y1, x, y = map(float, match)
                points = [[start_x, start_y], [x1, y1], [x, y]]
            
            # 'L' Line command (end point)
            line_matches = re.findall(r'L\s*([-\d.]+)[,\s]+([-\d.]+)', d_attr)
            for match in line_matches:
                x, y = map(float, match)
                points= [[start_x, start_y], [x, y]]
            
            paths_in_group.append(np.array(points))
        
        # Append group id and its paths with control points to the list
        groups_with_paths.append({'group_id': group_id, 'paths': paths_in_group})
    
    return groups_with_paths

# =========================
# ==== Bezier Sampling ====
# =========================
def cubic_bezier(P0, P1, P2, P3, t):
    """Calculate a point in the cubic Bezier curve at parameter t."""
    return (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3

# Quadratic Bezier Curve with three control points
def quadratic_bezier(P0, P1, P2, t):
    """Calculate a point on the quadratic Bézier curve at parameter t."""
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2

# Linear Bezier Curve with two control points (a straight line)
def linear_bezier(P0, P1, t):
    """Calculate a point on the linear Bézier curve (line) at parameter t."""
    return (1 - t) * P0 + t * P1

# Constant Bézier Curve with one control point (a single point)
def constant_bezier(P0, t):
    """Return a constant point for a constant Bézier curve."""
    return P0

def sample_point(num_cp, points, t):
    if num_cp == 4:
        point = cubic_bezier(points[0], points[1], points[2], points[3], t)
    elif num_cp == 3:
        point = quadratic_bezier(points[0], points[1], points[2], t)
    elif num_cp == 2:
        point = linear_bezier(points[0], points[1], t)
    elif num_cp == 1:
        point = constant_bezier(points[0], t)
    return point


def write_t_values(t_values_grid):
    t_values_txt = f"<t_values>"
    normalized_ts = []
    min_time = min(t_values_grid)
    max_time = max(t_values_grid)
    for t in t_values_grid:
        cur_n_t = (t - min_time) / (max_time - min_time) if max_time > min_time else 0.0
        normalized_ts.append(float(f"{cur_n_t:.2f}"))
        t_values_txt += f"{cur_n_t:.2f}, "
    t_values_txt = t_values_txt[:-2]
    t_values_txt += "</t_values>\n"
    return t_values_txt


def parse_xml_string_single_stroke(llm_output, res, stroke_counter):
    strokes_start_marker = f"<s{stroke_counter}>"
    strokes_end_marker = f"</s{stroke_counter}>"

    # Find the start and end indices of the JSON string
    start_index = llm_output.find(strokes_start_marker)
    if start_index != -1:
        # start_index += len(strokes_start_marker)  # Move past the marker
        end_index = llm_output.find(strokes_end_marker, start_index)
    else:
        return None  # XML markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    strokes_str = llm_output[start_index:end_index + len(strokes_end_marker)].strip()#[:-1]
    xml_str = f"<wrap>{strokes_str}</wrap>"
    # Parse the XML string
    root = ET.fromstring(xml_str)
    
    # Initialize lists to hold strokes and t_values
    # strokes_list = "[\n"
    # t_values_list = "[\n"
    
    # Iterate over all the strokes
    stroke = root.find(f"s{stroke_counter}")
    points_text = stroke.find('points').text

    # Extract t_values and convert them to float
    t_values_text = stroke.find('t_values').text

    # Append to the lists
    strokes_list = f"[{points_text}]"
    t_values_list = f"[{t_values_text}]"
    
    strokes_list = re.sub(r'\d+', lambda x: str(min(int(x.group()), res)), strokes_list)
    strokes_list = re.sub(r'\d+', lambda x: str(max(int(x.group()), 1)), strokes_list)
    
    return strokes_list, t_values_list

def bezier_point(P, t):
    """Calculate a point on the Bézier curve for a given t."""
    return (1-t)**3 * P[0] + 3*(1-t)**2 * t * P[1] + 3*(1-t) * t**2 * P[2] + t**3 * P[3]

def estimate_bezier_control_points_helper(sampled_points, t_values):
    n = len(sampled_points)
    
    if n == 1:
        # Linear interpolation: the control points are simply the two points
        print("sampled_points[0]", sampled_points[0])
        P0 = np.array(sampled_points[0])
        P1 = np.array(sampled_points[0]).astype(np.float64) + 0.0001
        return np.array([P0, P1])
        
    if n == 2:
        # Linear interpolation: the control points are simply the two points
        P0 = np.array(sampled_points[0])
        P1 = np.array(sampled_points[1])
        return np.array([P0, P1])

    if n > len(t_values):
        t_values = np.linespace(0,1,n)
    
    elif n == 3:
        # Quadratic Bézier curve: we need to solve for three control points
        A = np.zeros((n, 3))
        for i in range(n):
            t = t_values[i]
            A[i, 0] = (1-t)**2
            A[i, 1] = 2*(1-t)*t
            A[i, 2] = t**2
        
        # Points (flattened)
        B = np.array(sampled_points).reshape(-1, 2)  # Assuming 2D points
        
        # Solve the system (least squares)
        P = np.linalg.lstsq(A, B, rcond=None)[0]
        return P

    # Matrix A
    A = np.zeros((n, 4))
    for i in range(n):
        t = t_values[i]
        A[i, 0] = (1-t)**3
        A[i, 1] = 3*(1-t)**2 * t
        A[i, 2] = 3*(1-t) * t**2
        A[i, 3] = t**3
    
    # Points (flattened)
    B = np.array(sampled_points).reshape(-1, 2)  # Assuming 2D points
    
    # Solve the system (least squares)
    P = np.linalg.lstsq(A, B, rcond=None)[0]
    return P

    
def estimate_bezier_control_points( sampled_points, t_values):
    if len(sampled_points) != len(t_values):
        t_values = np.linspace(0,1, len(sampled_points))
    P = estimate_bezier_control_points_helper(sampled_points, t_values)

    if len(sampled_points) > 4:
        # Calculate the mean squared error between sampled points and the fitted Bézier curve.
        errors = []
        for i, t in enumerate(t_values):
            B_t = bezier_point(P, t)
            error = np.linalg.norm(B_t - sampled_points[i])
            errors.append(error)
        error = np.mean(errors)
        
        if error > 5 and len(sampled_points) >= 7:
            print(f"Fit error {error} is greater than tolerance. Splitting points and retrying...")
            mid = len(sampled_points) // 2
            left_sampled_points = sampled_points[:mid+1]
            right_sampled_points = sampled_points[mid:]
            left_t_values = np.array(t_values[:mid+1])
            right_t_values = np.array(t_values[mid:])

            if len(left_sampled_points) == 3: # this applies in case we have 7 points
                left_sampled_points.append(right_sampled_points[0])
                left_t_values.append(right_t_values[0])
                
            # Normalize t_values for each segment
            left_t_values = (left_t_values - left_t_values[0]) / (left_t_values[-1] - left_t_values[0])
            right_t_values = (right_t_values - right_t_values[0]) / (right_t_values[-1] - right_t_values[0])

            # Recursively fit curves to each segment
            P_left = estimate_bezier_control_points_helper(left_sampled_points, left_t_values)
            P_right = estimate_bezier_control_points_helper(right_sampled_points, right_t_values)
            P_right[0] = P_left[-1] # I added this to have the long strokes look more connected
            return [P_left, P_right]
    return [P]

def get_control_points(strokes_all, t_values_all, cells_to_pixels_map):
    net_points = []      
    for j in range(len(strokes_all)):
        sampled_cells = strokes_all[j]
        t_values = t_values_all[j]
        sampled_points = []
        for cell in sampled_cells:
            y,x = cells_to_pixels_map[cell]
            sampled_points.append([y,x])
        points_lst = estimate_bezier_control_points(sampled_points, t_values)
        net_points.append(points_lst)
    return net_points

def get_control_points_single_stroke(strokes_all, t_values_all, cells_to_pixels_map):
    sampled_points = []
    for cell in strokes_all:
        y,x = cells_to_pixels_map[cell]
        sampled_points.append([y,x])
    points_lst = estimate_bezier_control_points(sampled_points, t_values_all)
    return points_lst


def create_svg_path_data(control_points):
    # Start the path with 'M' for the first point
    # print("control_points", control_points[0])
    path_data = 'M ' + np.array2string(np.array(control_points[0]), formatter={'float_kind':lambda x: "%.2f" % x}, separator=' ')[1:-1]    
    # Add 'L' for each subsequent point

    # check if point
    if len(control_points) == 1:
        path_data += ' '
    # check if line
    elif len(control_points) == 2:
        path_data += ' L '
    # check if quadratic
    elif len(control_points) == 3:
        path_data += ' Q '
    # check if cubic
    elif len(control_points) == 4:
        path_data += ' C '
    
    # path_data += ' C '
    for point in control_points[1:]:
        # print("pt", point[0], point[1])
        path_data += str(point[0]) + " " + str(point[1]) + " "
    
    # Return the complete 'd' attribute string
    return path_data

def get_stroke_color(stroke_counter):
    if stroke_counter % 2 == 0:
        return "red"
    return "blue"

def format_svg_single_stroke(group, dim, stroke_width, stroke_counter, stroke_color="black"):
    # stroke_color = get_stroke_color(stroke_counter)
    svg_width, svg_height = dim
    sketch_text_svg = ""
    # sketch_text_svg = f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">\n"""        
    gropu_text = f"""<g id="s{stroke_counter}" stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" stroke-linecap="round">\n"""
    for sub_path_cp in group:  #sometimes 1 or 2 
        path_data = create_svg_path_data(sub_path_cp)
        gropu_text += f"""<path d="{path_data}"/>\n"""
    gropu_text += "</g>\n"
    sketch_text_svg += gropu_text
    # sketch_text_svg += "</svg>"
    return sketch_text_svg

def format_svg(all_control_points, dim, stroke_width):
    svg_width, svg_height = dim
    sketch_text_svg = f"""<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">\n"""        
    for i, path in enumerate(all_control_points):
        gropu_text = f"""<g id="s{i + 1}" stroke="black" stroke-width="{stroke_width}" fill="none" stroke-linecap="round">\n"""
        for sub_path_cp in path:  #sometimes 1 or 2 
            path_data = create_svg_path_data(sub_path_cp)
            gropu_text += f"""<path d="{path_data}"/>\n"""
        gropu_text += "</g>\n"
        sketch_text_svg += gropu_text
    sketch_text_svg += "</svg>"
    return sketch_text_svg

def get_cur_stroke_text(stroke_counter, llm_output):
    start_marker = f"<s{stroke_counter}>"
    end_marker = f"</s{stroke_counter}>"

    # Find the start and end indices of the JSON string
    start_index = llm_output.find(start_marker)
    if start_index != -1:
        # start_index += len(strokes_start_marker)  # Move past the marker
        end_index = llm_output.find(end_marker, start_index)
    else:
        return ""  # XML markers not found

    if end_index == -1:
        return ""  # End marker not found

    # Extract the JSON string
    strokes_str = llm_output[start_index:end_index + len(end_marker)].strip()#[:-1]
    return strokes_str


# Note that this parse only the *first* part in the text in which you have the <strokes> </strokes> tags.
def parse_xml_string(llm_output, res):

    strokes_start_marker = "<strokes>"
    strokes_end_marker = "</strokes>"

    # Find the start and end indices of the JSON string
    start_index = llm_output.find(strokes_start_marker)
    if start_index != -1:
        # start_index += len(strokes_start_marker)  # Move past the marker
        end_index = llm_output.find(strokes_end_marker, start_index)
    else:
        return None  # XML markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    strokes_str = llm_output[start_index:end_index + len(strokes_end_marker)].strip()#[:-1]
    xml_str = f"<wrap>{strokes_str}</wrap>"
    # Parse the XML string
    root = ET.fromstring(xml_str)
    
    # Initialize lists to hold strokes and t_values
    strokes_list = "[\n"
    t_values_list = "[\n"
    
    # Iterate over all the strokes
    for stroke in root.find('strokes'):
        # Extract points and clean them up
        points_text = stroke.find('points').text
    
        # Extract t_values and convert them to float
        t_values_text = stroke.find('t_values').text
    
        # Append to the lists
        strokes_list += f"[{points_text}],\n"
        t_values_list += f"[{t_values_text}],\n"
    
    strokes_list = re.sub(r'\d+', lambda x: str(min(int(x.group()), res)), strokes_list)
    strokes_list = re.sub(r'\d+', lambda x: str(max(int(x.group()), 1)), strokes_list)
    
    strokes_list += "]"
    t_values_list += "]"
    return strokes_list, t_values_list

def get_strokes_text(llm_output):
    strokes_start_marker = "<strokes>"
    strokes_end_marker = "</strokes>"

    # Find the start and end indices of the JSON string
    start_index = llm_output.find(strokes_start_marker)
    if start_index != -1:
        # start_index += len(strokes_start_marker)  # Move past the marker
        end_index = llm_output.find(strokes_end_marker, start_index)
    else:
        return None  # XML markers not found

    if end_index == -1:
        return None  # End marker not found

    # Extract the JSON string
    strokes_str = llm_output[start_index:end_index + len(strokes_end_marker)].strip()#[:-1]
    return strokes_str