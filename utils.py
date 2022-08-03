from dgllife.data import BACE
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import SMILESToBigraph
from dgllife.model.pretrain.moleculenet import create_bace_model

from rdkit import Chem

node_featurizer = CanonicalAtomFeaturizer()
s2g = SMILESToBigraph(add_self_loop=True, node_featurizer=node_featurizer, edge_featurizer=None)
model = create_bace_model("GCN_canonical_BACE")
model.eval()

def predict_bace(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return model(s2g(smiles), feats=node_featurizer(mol)["h"]).item()
