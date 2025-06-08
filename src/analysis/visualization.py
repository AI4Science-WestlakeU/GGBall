import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors, Crippen, QED
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import imageio
import networkx as nx
import numpy as np
import rdkit.Chem
import wandb
import matplotlib.pyplot as plt

from rdkit import DataStructs
from PIL import Image, ImageDraw

def tanimoto(mol1, mol2, radius=2, nBits=2048):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

class MolecularVisualization:
    def __init__(self, remove_h, dataset_infos):
        self.remove_h = remove_h
        self.dataset_infos = dataset_infos

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        # dictionary to map integer value to the char of atom
        atom_decoder = self.dataset_infos.atom_decoder

        # create empty editable mol object
        mol = Chem.RWMol()

        # add atoms to mol and keep track of index
        node_to_idx = {}
        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            a = Chem.Atom(atom_decoder[int(node_list[i])])
            molIdx = mol.AddAtom(a)
            node_to_idx[i] = molIdx

        for ix, row in enumerate(adjacency_matrix):
            for iy, bond in enumerate(row):
                # only traverse half the symmetric matrix
                if iy <= ix:
                    continue
                if bond == 1:
                    bond_type = Chem.rdchem.BondType.SINGLE
                elif bond == 2:
                    bond_type = Chem.rdchem.BondType.DOUBLE
                elif bond == 3:
                    bond_type = Chem.rdchem.BondType.TRIPLE
                elif bond == 4:
                    bond_type = Chem.rdchem.BondType.AROMATIC
                else:
                    continue
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

        try:
            mol = mol.GetMol()
        except rdkit.Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    def visualize(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)
        
        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(path, 'molecule_{}.png'.format(i))
            mol = self.mol_from_graphs(molecules[i][0].numpy(), molecules[i][1].numpy())
            try:
                Draw.MolToFile(mol, file_path)
                if wandb.run and log is not None:
                    print(f"Saving {file_path} to wandb")
                    wandb.log({log: wandb.Image(file_path)}, commit=True)
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")

    # def visualize_interpolate(self, path: str, molecules: list, num_molecules_to_visualize: int, log='graph'):

    def calc_props(self,mol):
        """返回常用性质字典"""
        valid = self.is_valid_mol(mol)
        if not valid:
            return {
                "logP": 0,
                "QED": 0,
                "valid": valid,
                "formula": "N/A",
                "rings": "N/A",
            }
        return {
            # "MW": Descriptors.MolWt(mol),
            "logP": Crippen.MolLogP(mol),
            "QED": QED.qed(mol),
            "formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
            "rings": mol.GetRingInfo().NumRings(),
            "valid": valid
        }

    def is_valid_mol(self, mol: Chem.Mol) -> bool:
        """
        如果 mol 为 None，或者在 Sanitize 过程中抛异常，则认为无效。
        返回 True / False
        """
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol)              # 会做价键、芳香性、价态等一系列合法性检查
            return True
        except Chem.rdchem.KekulizeException:
            # 芳香化失败
            return False
        except ValueError:
            # 其他 Sanitize 错误
            return False
        
    def visualize_interpolation_strip(
        self,
        molecule_list,
        path: str,
        img_size=(250, 250),             # 单个分子尺寸
        log_tag: str = "interp_strip",   # wandb 的 key
        alpha_list=None                  # 对应的 t 值，可手动传入
    ):
        """
        molecule_list: [(atom_types, edge_types), ...] ⟂ len=11
        mol_from_graphs: callable(atom_types, edge_types) -> RDKit Mol
        path: 保存文件夹
        """
        print(f"start visualize interpolation strip")
        if alpha_list is None:                       # 默认均匀 0.0 ~ 1.0
            steps = len(molecule_list) - 1
            alpha_list = [i / steps for i in range(steps + 1)]

        # 1. 转 RDKit Mol 并准备 legend
        mols, legends = [], []
        for (atom_types, edge_types), t in zip(molecule_list, alpha_list):
            print(f"visualize interpolation strip {t}")
            mol = self.mol_from_graphs(atom_types.numpy(), edge_types.numpy())
            mols.append(mol)
            props = self.calc_props(mol)
            legends.append(f"valid: {props['valid']}\nformula: {props['formula']}\nrings: {props['rings']}\n")
            # logP = {props['logP']:.2f}, QED = {props['QED']:.2f}
            

        # 2. 生成单行网格图
        strip_img = Draw.MolsToGridImage(
            mols,
            molsPerRow=len(mols),        # 单行
            subImgSize=img_size,
            legends=legends,
            useSVG=False,
            returnPNG=True               # 直接拿到 PNG bytes
        )

        # 3. 保存到本地
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "interpolation_strip.png")
        with open(file_path, "wb") as f:
            f.write(strip_img)

        print(f"Saved strip to {file_path}")

        # 4. 上传 wandb（可选）
        if wandb.run:
            wandb.log({log_tag: wandb.Image(file_path)}, commit=True)

        return file_path  # 方便后续调用

    def mol_to_pil_with_caption_bg(self, mol, caption):
        # ① 先画分子
        mol_img = Draw.MolToImage(mol, size=(250, 200)).convert("RGBA")

        # ② 判断合法性后选底色（RGBA）
        if self.is_valid_mol(mol):
            bg = (230, 255, 230, 160)   # very light green, 60 % alpha
        else:
            bg = (255, 230, 230, 160)   # very light red
        w, h = mol_img.size
        h_cap = 20

        # ③ 造底布 & 贴图
        canvas = Image.new("RGBA", (w, h + h_cap), bg)
        canvas.paste(mol_img, (0, 0), mol_img)

        # ④ 写标题
        draw = ImageDraw.Draw(canvas)
        draw.text((5, h + 2), caption, fill=(0, 0, 0, 255))
        return canvas
    def visualize_interpolation_strip2(self, molecules, save_path, log_tag=None):
        """
        molecules: RDKit Mol list，11 个
        """
        imgs = []
        for atom_types, edge_types in molecules:
            mol = self.mol_from_graphs(atom_types.numpy(), edge_types.numpy())
            props = self.calc_props(mol)
            formula = props['formula']
            rings   = props['rings']
            cap     = f"formula={formula} | rings={rings}"
            imgs.append(self.mol_to_pil_with_caption_bg(mol, cap))

        # -------- 横向拼接 --------
        widths, heights = zip(*(im.size for im in imgs))
        strip = Image.new('RGB', (sum(widths), max(heights)), 'white')
        x_off = 0
        for im in imgs:
            strip.paste(im, (x_off, 0))
            x_off += im.width

        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, f"{log_tag}.png")
        strip.save(save_path)

        # (可选) 传给 wandb
        # if wandb.run:
        #     wandb.log({"interpolation_strip": wandb.Image(save_path)}, commit=True)

    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        RDLogger.DisableLog('rdApp.*')
        # convert graphs to the rdkit molecules
        mols = [self.mol_from_graphs(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]

        # find the coordinates of atoms in the final molecule
        final_molecule = mols[-1]
        AllChem.Compute2DCoords(final_molecule)

        coords = []
        for i, atom in enumerate(final_molecule.GetAtoms()):
            positions = final_molecule.GetConformer().GetAtomPosition(i)
            coords.append((positions.x, positions.y, positions.z))

        # align all the molecules
        for i, mol in enumerate(mols):
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            for j, atom in enumerate(mol.GetAtoms()):
                x, y, z = coords[j]
                conf.SetAtomPosition(j, Point3D(x, y, z))

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            Draw.MolToFile(mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}")
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)

        if wandb.run:
            print(f"Saving {gif_path} to wandb")
            wandb.log({"chain": wandb.Video(gif_path, fps=5, format="gif")}, commit=True)

        # draw grid image
        try:
            img = Draw.MolsToGridImage(mols, molsPerRow=10, subImgSize=(200, 200))
            img.save(os.path.join(path, '{}_grid_image.png'.format(path.split('/')[-1])))
        except Chem.rdchem.KekulizeException:
            print("Can't kekulize molecule")
        return mols


class NonMolecularVisualization:
    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            graph.add_edge(edge[0], edge[1], color=float(edge_type), weight=3 * edge_type)

        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        plt.figure()
        nx.draw(graph, pos, font_size=5, node_size=node_size, with_labels=False, node_color=U[:, 1],
                cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph'):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            if wandb.run and log is not None:
                wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        # convert graphs to networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
        if wandb.run:
            wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
