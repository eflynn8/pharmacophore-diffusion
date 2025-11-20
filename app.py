import streamlit as st
import os
import subprocess
import tempfile
import py3Dmol
import streamlit.components.v1 as components
import json

# Set page configuration
st.set_page_config(layout="wide", page_title="PharmacoForge")

st.markdown('''<style> 
            .stAppHeader {
            background: rgba(255, 255, 255, 0.5);
            }
            section.stSidebar {
              background: rgba(240, 242, 246,0.5)
            }
            iframe.stIFrame {
                position: fixed;
                top: 0;
                left: 0;
                height: 100dvh;
                width: 100%;
                margin: 0;
                padding: 0;
            }            
            .block-container {
                padding-top: 2.5rem;
                padding-bottom: 0rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }            
            </style>
''',unsafe_allow_html=True)

# Function to parse XYZ file (pharmacophores)
# assumes only one of each size, returns dictionary indexed by size
def parse_xyz(file_path):
    radius = {
        'P': 1,    # Aromatic
        'S': 0.9,  # Hydrogen Donor
        'F': 1,    # Hydrogen Acceptor
        'C': 0.9,  # Hydrophobic
        'O': 0.8,  # Negative Ion
        'N': 0.8   # Positive Ion
    }
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        all_pharmacophores = {}
        while lines:
            line = lines[0].strip()
            if not line: 
                lines = lines[1:]
                continue
                
            try:
                num_feats = int(line)
            except ValueError:
                lines = lines[1:]
                continue
                
            pharmacophore = []
            chunk = lines[1:num_feats+1]
            for feat_line in chunk:
                parts = feat_line.split()
                if len(parts) >= 4:
                    pharmacophore.append({
                        'name': parts[0],
                        'coords': {'x': float(parts[1]), 'y': float(parts[2]), 'z': float(parts[3])},
                        'radius': radius.get(parts[0], 1.0)
                    })
            all_pharmacophores[num_feats] = pharmacophore
            lines = lines[num_feats+1:]
            
        return all_pharmacophores
    except Exception as e:
        st.error(f"Error parsing XYZ file: {e}")
        return []


with st.sidebar:

    st.logo("""<svg width="586" height="86" viewBox="0 0 586 86" xmlns="http://www.w3.org/2000/svg">
  <text
    x="0"
    y="64"
    fill="#202328"
    font-family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
    font-size="64"
    font-weight="600">
    PharmacoForge
  </text>
</svg>
""")
    uploaded_receptor = st.file_uploader("Upload Receptor (PDB)", type=['pdb'])

    pocket_method = st.radio("Pocket Definition", ["Reference Ligand", "Residue List"]) 

    uploaded_ligand = None
    residue_list_str = ""

    if pocket_method == "Reference Ligand":
        uploaded_ligand = st.file_uploader("Upload Reference Ligand (SDF)", type=['sdf'])
    else:
        residue_list_str = st.text_input("Residue List (e.g. A:1 A:2)", help="Chain:ResidueIndex")
        
    seed = st.number_input("Random Seed",value=42)
    generate_btn = st.button("Generate Pharmacophore", type="primary")


if generate_btn:
    if uploaded_receptor is None:
        st.error("Please upload a receptor file.")    
    elif pocket_method == "Reference Ligand" and not uploaded_ligand:
        st.error("Please upload a reference ligand.")
    elif pocket_method == "Residue List" and not residue_list_str:
        st.error("Please specify the residue list.")    
    else:
        with st.spinner("Running model...",show_time=True,width='stretch'):
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Save inputs to temp files
                receptor_path = os.path.join(tmp_dir, uploaded_receptor.name)
                with open(receptor_path, "wb") as f:
                    f.write(uploaded_receptor.getbuffer())
                
                # Construct command
                cmd = ["python", "generate_pharmacophores.py", receptor_path]
                cmd.extend(["--model_dir", "model_dir"])
                cmd.extend(["--samples_per_pocket", "6"])
                cmd.extend(["--seed",str(seed)])
                cmd.append("--pharm_sizes")
                cmd.extend(map(str,[3,4,5,6,7,8]))
                
                ligand = None
                if pocket_method == "Reference Ligand":
                    ligand_path = os.path.join(tmp_dir, uploaded_ligand.name)
                    with open(ligand_path, "wb") as f:
                        ligand = uploaded_ligand.read()
                        f.write(ligand)
                        ligand = ligand.decode()
                    cmd.extend(["--ref_ligand_file", ligand_path, "--use_ref_lig_com"])
                else:
                    cmd.append("--residue_list")
                    cmd.extend(residue_list_str.split())
                
                # Execute generation script
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Output expected at: generated_pharms/<receptor_name>/pharms.xyz
                    receptor_name = os.path.splitext(uploaded_receptor.name)[0]
                    output_path = os.path.join("generated_pharms", receptor_name, "pharms.xyz")
                    
                    if os.path.exists(output_path):                        
                        # Visualize results
                        pharmacophores = parse_xyz(output_path)
                        
                        view = py3Dmol.view(width="100%", height="100%")
                        

                        # Add pharmacophores
                        cmap = {'P': 'purple', 'S': 'white', 'F': 'orange', 'C': 'green', 'O': 'red', 'N': 'blue'}
                        
                        pharm = pharmacophores[8]
                        for feat in pharm:
                            view.addSphere({
                                'center': feat['coords'],
                                'radius': feat['radius'],
                                'color': cmap.get(feat['name'], 'gray'),
                                'wireframe': True
                            })
                        
                        receptor = uploaded_receptor.read().decode('utf-8')
                        name_mapping = {
                            'P': 'Aromatic',
                            'S': 'HydrogenDonor',
                            'F': 'HydrogenAcceptor',
                            'C': 'Hydrophobic',
                            'O': 'NegativeIon',
                            'N': 'PositiveIon'
                        }
                        #precomptue pharmit json
                        pharmit_points = {}
                        for i,pharmacophore in pharmacophores.items():
                            points = []
                            for feat in pharmacophore:
                                points.append({
                                    'enabled': True,
                                    'name': name_mapping[feat['name']],
                                    'radius': feat['radius'],
                                    'x': feat['coords']['x'],
                                    'y': feat['coords']['y'],
                                    'z': feat['coords']['z']
                                })
                            pharmit_points[i]={'points': points}
                        pharmit_structs = {'recname': uploaded_receptor.name,
                                      'receptor': receptor}
                        if ligand is not None:
                            pharmit_structs['ligand'] = ligand
                            pharmit_structs['ligname'] = uploaded_ligand.name

                        view.zoomTo()

                        # Add receptor
                        uploaded_receptor.seek(0)
                        view.addModel(receptor, 'pdb')
                        view.setStyle({'cartoon': {'color': 'white'},
                                        'stick': {'colorscheme': 'whiteCarbon', 'radius': 0.15}})
                        view.addSurface(py3Dmol.VDW,{'opacity': 0.7, 'colorscheme': 'whiteCarbon'})

                        view.addModel(ligand)
                        view.setStyle({'model':1},'stick')
                        components.html('''<style>
                                        html, body, iframe {
                                        height: 100%;
                                        width: 100%;
                                        margin: 0;
                                        padding: 0;
                                        }
                                        #container {
                                        height: 100%;
                                        width: 100%;
                                        background: #eef2ff;
                                        }
                                        </style>
                                        '''+ view.write_html(),
                                        height="100%")
                        uid = view.uniqueid

                        with st.container(key='controls',gap=None):
                            st.markdown('''<style> 
                                .st-key-controls iframe.stIFrame {
                                    position: relative;
                                    height: auto;
                                    width: auto;
                                }
                                </style>
                            ''',unsafe_allow_html=True)
                            components.html('''
                            <style>
                                #pharm-container {
                                    background: rgba(255, 255, 255, 0.5);
                                    padding: 10px 15px;
                                    border-radius: 4px;
                                    display: inline-block;
                                    font-family: sans-serif;
                                }
                                label { margin-right: 10px; }
                                select {
                                    background: rgba(255, 255, 255, 0.5);
                                    border: 1px solid #ccc;
                                    border-radius: 4px;
                                    padding: 5px 8px;
                                }
                                select:hover {
                                    background: rgba(255, 255, 255, 0.7);
                                }
                                button {
                                    background: rgba(255, 255, 255, 0.5);
                                    border: 1px solid #ccc;
                                    border-radius: 4px;
                                    padding: 6px 12px;
                                    cursor: pointer;
                                    font-family: sans-serif;
                                }
                                button:hover {
                                    background: rgba(255, 255, 255, 0.7);
                                }
                            </style>
                            <script>
                                    var pharmacophores = PHARMACOPHORES;
                                    var points = POINTS;
                                    var structs = STRUCTS;
                                    var cmap = {'P': 'purple', 'S': 'white', 'F': 'orange', 'C': 'green', 'O': 'red', 'N': 'blue'}

                                    function display_pharm(n) {
                                        let viewer = window.parent.frames[0].viewer_UNIQUEID;
                                        viewer.removeAllShapes();
                                        n = parseInt(n)
                                        let pharm = pharmacophores[n];
                                        for(const feat of pharm) {
                                            viewer.addSphere({
                                                'center': feat['coords'],
                                                'radius': feat['radius'],
                                                'color': cmap[feat['name']],
                                                'wireframe': true
                                            });
                                        }
                                        viewer.render();
                                    };
                                    //object for sending messages to a window, but only after we receive an ack
                                    function Message(data, w, dest) {
                                        var curWindow = w;
                                        var curDest = dest;
                                        var curMsg = data;
                                        var isAcked = 0;
                                        function receiveMessage(event) {
                                            if (event.data == "ack2") {
                                                isAcked = 1;
                                            }
                                        }

                                        function check() {
                                            if (isAcked) {
                                                curWindow.postMessage(curMsg, curDest);
                                                curDest = "";
                                                curMsg = "";
                                                curWindow = null;
                                                isAcked = 0;
                                                window.removeEventListener("message", receiveMessage);
                                            } else if (curWindow) {
                                                curWindow.postMessage("ack", curDest);
                                                setTimeout(check, 250);
                                            }
                                        }

                                        window.addEventListener("message", receiveMessage);
                                        w.postMessage("ack", dest);
                                        setTimeout(check, 250);
                                    }
                                                
                                    function send_pharmit() {
                                        let viewer = window.parent.frames[0].viewer_UNIQUEID;                                            
                                        const size = parseInt(document.getElementById('pharm-select').value);
                                        let pharm = {...points[size], ...structs};
                                        pharm['view'] = viewer.getView();
                                        var win = window.open("http://pharmit.csb.pitt.edu/search.html");
                                        var msg = new Message(JSON.stringify(pharm), win, '*');
                                                                                    
                                    };
                            </script>
                            <div id="pharm-container">
                                <label for="pharm-select">Size:</label>
                                <select id="pharm-select" onchange="display_pharm(this.value);">
                                    <option value="3">3</option>
                                    <option value="4">4</option>
                                    <option value="5">5</option>
                                    <option value="6">6</option>
                                    <option value="7">7</option>
                                    <option value="8" selected>8</option>
                                </select>
                                <button onclick="send_pharmit()">Send to Pharmit</button>
                            </div>
                            '''.replace('UNIQUEID',str(uid))
                            .replace('PHARMACOPHORES',json.dumps(pharmacophores))
                            .replace('POINTS',json.dumps(pharmit_points))
                            .replace('STRUCTS',json.dumps(pharmit_structs)))
                    else:
                        st.error("Output file not generated.")
                        st.text(f"Expected output at: {output_path}")
                else:
                    st.error("Model execution failed. Check input formatting.")
                    st.text("Standard Error:")
                    st.code(result.stderr)


                    


