# -*- coding: utf-8 -*-

import json
from typing import List, Dict
from pathlib import Path
from llama_index.core.schema import TextNode

# Load jason file
def load_json_files(data_dir:str) -> List[Dict]:
    """Load and Validate the json files in data_dir"""
    json_files = list(Path(data_dir).glob('*.json'))
    assert json_files, f"Can't find JSON files in {data_dir}"

    law_Data = []
    for json_file in json_files:
        with open(json_file,'r', encoding='utf-8') as f:
            try:
                data = json.load(f)

                if not isinstance(data, dict):
                    raise ValueError(f"{json_file.name} is not a dict")
                for k,v in data.items():
                    if not isinstance(v, str):
                        raise ValueError(f"The value for key:{k} in {json_file.name} is not a str")
                law_Data.extend({
                    "content":data,
                    "metadata":{"source":json_file.name}
                })
            except Exception as e:
                raise RuntimeError(f"Loading {json_file.name} failed: {str(e)}")

    print(f"Loaded {len(law_Data)} law data")
    return law_Data


def create_nodes(raw_Data: List[Dict]) -> List[TextNode]:

    nodes = []
    for data in raw_Data:
        law_Dict = data['content']
        source_file = data['metadata']['source']

        for full_title, content in law_Dict.items():
            node_id = f"{source_file}::{full_title}"

            parts = full_title.split(" ",1)
            section_num = parts[0] if len(parts) > 0 else 'Unknow Section'
            section_name = parts[1] if len(parts) > 1 else 'Unknow Section Name'

            node = TextNode(
                text=content,
                id=node_id,
                metadata={
                    "Section_Name": section_name,
                    "Section_Num": section_num,
                    "source":source_file,
                    "content_type":"Law Act"
                }
            )
            nodes.append(node)
    print(f"Created {len(nodes)} nodes (ID Example: {nodes[0].id})")
    return nodes
