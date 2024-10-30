import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np

# Load the data
data = pd.read_csv('z_score_combined_score.csv')


def create_xml_prompt(row):
    # Create the root element
    essay = ET.Element("essay")
    
    # Create sub-elements for each column
    for col in row.index:
        value = row[col]
        
        # Convert integers to null and strings to NA as specified
        if pd.isnull(value):
            continue
            # value_element = ET.SubElement(essay, col)
            # value_element.text = "NA"  # For string columns
        elif isinstance(value, (int, float)):
            value_element = ET.SubElement(essay, col)
            value_element.text = str(value) if value is not None else "null"  # For integer/float
        else:
            value_element = ET.SubElement(essay, col)
            value_element.text = str(value)  # Handle other types
            
    # Convert the XML element tree to a string
    return ET.tostring(essay, encoding='unicode')


# generate Prompts



# Generate XML for each row
data['xml_prompt'] = data.apply(create_xml_prompt, axis=1)

# Display the first few XML prompts
print(data['xml_prompt'])
print(data['xml_prompt'][1])
