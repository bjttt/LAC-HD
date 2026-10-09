"""CPU tensor tests; no Transformers install, model download or GPU needed."""
from __future__ import annotations
import gc
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Code'))
from confidence_head import LayerAttentionConfidenceHead
from confidence_head_factory import build_head
from llama7b_confidence_head import Llama7BConfidenceHead


class Llama7BHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.head=build_head({'model_type':'llama','head_type':'base','dropout':0.0},4096,32)
        cls.head.eval()

    def test_real_7b_shapes_and_training_gradients(self):
        self.assertIsInstance(self.head,Llama7BConfidenceHead)
        states=torch.randn(1,3,32,4096)
        mask=torch.tensor([[1,1,0]])
        logits,weights=self.head(states,mask)
        self.assertEqual(tuple(logits.shape),(1,3))
        self.assertEqual(tuple(weights.shape),(1,3,32))
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(1,3),atol=1e-6))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[:,:2],torch.tensor([[0.,1.]]))
        loss.backward()
        for module in [self.head.layer_attn[0],self.head.context_conv,self.head.mlp[-1]]:
            self.assertIsNotNone(module.weight.grad)
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()),0.)
        self.head.zero_grad(set_to_none=True)

    def test_padded_states_do_not_change_real_token_scores(self):
        x=torch.randn(1,3,32,4096)
        changed=x.clone();changed[:,2]=100.*torch.randn_like(changed[:,2])
        mask=torch.tensor([[1,1,0]])
        with torch.no_grad():
            a,_=self.head(x,mask);b,_=self.head(changed,mask)
        self.assertTrue(torch.allclose(a[:,:2],b[:,:2],atol=1e-6))

    def test_state_dict_compatible_with_shared_head(self):
        # Avoid a second large model allocation: compare architecture keys/shapes
        # to a small shared head, then round-trip the real head state on disk-free IO.
        small=LayerAttentionConfidenceHead(hidden_size=8,num_layers=2,attn_hidden=4,mlp_hidden1=8,mlp_hidden2=4)
        self.assertEqual(set(small.state_dict()),set(self.head.state_dict()))
        del small
        buffer=io.BytesIO()
        torch.save({'head_state_dict':self.head.state_dict()},buffer)
        buffer.seek(0)
        restored=torch.load(buffer,map_location='cpu',weights_only=True)['head_state_dict']
        self.head.load_state_dict(restored,strict=True)
        self.assertTrue(torch.equal(restored['mlp.6.weight'],self.head.mlp[-1].weight))
        del restored,buffer
        gc.collect()

    def test_wrong_backbone_or_tensor_shape_is_rejected(self):
        correct=SimpleNamespace(model_type='llama',hidden_size=4096,num_hidden_layers=32,vocab_size=32000)
        Llama7BConfidenceHead.validate_model_config(correct)
        for key,value in [('model_type','qwen3'),('num_hidden_layers',36),('hidden_size',5120),('vocab_size',128256)]:
            wrong=SimpleNamespace(**vars(correct));setattr(wrong,key,value)
            with self.assertRaises(ValueError): Llama7BConfidenceHead.validate_model_config(wrong)
        with self.assertRaises(ValueError): self.head(torch.zeros(1,2,36,4096))
        with self.assertRaises(ValueError): self.head(torch.zeros(1,2,32,4096),torch.ones(1,3))

    def test_other_backbones_and_structural_controls_keep_shared_factory(self):
        for kind in ['base','no_conv','deep_mlp','last_layer_linear','mean_layer_mlp']:
            head=build_head({'model_type':'qwen3','head_type':kind,'attn_hidden':4,
                             'mlp_hidden1':8,'mlp_hidden2':4,'mlp_hidden3':4,
                             'mlp_hidden4':2,'dropout':0.0},8,3)
            scores,weights=head(torch.randn(1,2,3,8),torch.ones(1,2))
            self.assertEqual(tuple(scores.shape),(1,2))
            self.assertEqual(tuple(weights.shape),(1,2,3))
        for kind in ['no_conv','deep_mlp','last_layer_linear','mean_layer_mlp']:
            head=build_head({'model_type':'llama','head_type':kind,'attn_hidden':4,
                             'mlp_hidden1':8,'mlp_hidden2':4,'mlp_hidden3':4,
                             'mlp_hidden4':2,'dropout':0.0,'no_context_conv':True},4096,32)
            with torch.no_grad(): scores,weights=head(torch.randn(1,2,32,4096),torch.ones(1,2))
            self.assertEqual(tuple(scores.shape),(1,2))
            self.assertEqual(tuple(weights.shape),(1,2,32))
            del head
            gc.collect()


if __name__=='__main__':
    unittest.main()
