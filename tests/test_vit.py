import torch
import unittest

from class_token import ClassToken
from encoder import Encoder
from encoder_layer import EncoderLayer
from layer_normalization import LayerNormalization
from mlp_layer import MLPLayer
from multi_headed_attention import MultiHeadedAttention
from patch_embeddings import PatchEmbedding
from positional_encoding import PositionalEncoding
from vit import ViT


class TestViT(unittest.TestCase):
    def setUp(self):
        # default ViT parameters used for the unittests
        self.dropout_prob = 0.1
        self.batch_size = 8
        self.input_ch = 3
        self.input_h = 224
        self.input_w = 224
        self.num_classes = 3
        self.patch_size = 16
        self.embedding_dim = 768
        self.num_heads = 8
        self.mlp_neurons = 2048
        self.num_layers = 6
        self.num_patches = (self.input_h // self.patch_size) * (self.input_w // self.patch_size)

        # input and various ViT layers to test
        self.test_input = torch.randn(self.batch_size, self.input_ch, self.input_h, self.input_w)
        self.test_patch_embedding = PatchEmbedding(in_channels=self.input_ch,
                                                   patch_size=self.patch_size,
                                                   embedding_dim=self.embedding_dim)
        self.test_class_token = ClassToken(batch_size=self.batch_size,
                                           embedding_dim=self.embedding_dim)
        self.test_positional_encoding = PositionalEncoding(num_patches=self.num_patches,
                                                           dropout_prob=self.dropout_prob,
                                                           embedding_dim=self.embedding_dim)
        self.test_multi_head_attention = MultiHeadedAttention(embed_size=self.embedding_dim,
                                                              num_heads=self.num_heads,
                                                              dropout_prob=self.dropout_prob)
        self.test_layer_norm = LayerNormalization(parameters_shape=[self.embedding_dim])
        self.test_mlp = MLPLayer(embed_size=self.embedding_dim,
                                 hidden_size=self.mlp_neurons,
                                 dropout_prob=self.dropout_prob)
        self.test_encoder_layer = EncoderLayer(embed_size=self.embedding_dim,
                                               num_heads=self.num_heads,
                                               mlp_hidden_size=self.mlp_neurons,
                                               dropout_prob=self.dropout_prob)
        self.test_encoder = Encoder(embed_size=self.embedding_dim,
                                    num_heads=self.num_heads,
                                    mlp_hidden_size=self.mlp_neurons,
                                    dropout_prob=self.dropout_prob,
                                    num_layers=self.num_layers)

        self.test_vit = ViT(batch_size=self.batch_size,
                            in_channels=self.input_ch,
                            patch_size=self.patch_size,
                            embedding_dim=self.embedding_dim,
                            num_patches=self.num_patches,
                            dropout_prob=self.dropout_prob,
                            num_heads=self.num_heads,
                            mlp_hidden_size=self.mlp_neurons,
                            num_layers=self.num_layers,
                            num_classes=self.num_classes)

    def test_patch_embeddings(self):
        test_output = self.test_patch_embedding(self.test_input)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_patches, self.embedding_dim]))

    def test_class_token(self):
        test_output = self.test_patch_embedding(self.test_input)
        test_output = self.test_class_token(test_output)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_patches + 1, self.embedding_dim]))

    def test_positional_encoding(self):
        test_output = self.test_patch_embedding(self.test_input)
        test_output = self.test_class_token(test_output)
        test_output = self.test_positional_encoding(test_output)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_patches + 1, self.embedding_dim]))

    def test_multi_headed_attention(self):
        test_output = self.test_patch_embedding(self.test_input)
        test_output = self.test_class_token(test_output)
        test_output = self.test_positional_encoding(test_output)
        test_output = self.test_multi_head_attention(test_output)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_patches + 1, self.embedding_dim]))

    def test_layer_normalization(self):
        test_output = self.test_patch_embedding(self.test_input)
        test_output = self.test_class_token(test_output)
        test_output = self.test_positional_encoding(test_output)
        test_output = self.test_multi_head_attention(test_output)
        test_output = self.test_layer_norm(test_output)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_patches + 1, self.embedding_dim]))

    def test_mlp_layer(self):
        test_output = self.test_patch_embedding(self.test_input)
        test_output = self.test_class_token(test_output)
        test_output = self.test_positional_encoding(test_output)
        test_output = self.test_multi_head_attention(test_output)
        test_output = self.test_layer_norm(test_output)
        test_output = self.test_mlp(test_output)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_patches + 1, self.embedding_dim]))

    def test_encoder_layer(self):
        test_output = self.test_patch_embedding(self.test_input)
        test_output = self.test_class_token(test_output)
        test_output = self.test_positional_encoding(test_output)
        test_output = self.test_encoder_layer(test_output)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_patches + 1, self.embedding_dim]))

    def test_encoder(self):
        test_output = self.test_patch_embedding(self.test_input)
        test_output = self.test_class_token(test_output)
        test_output = self.test_positional_encoding(test_output)
        test_output = self.test_encoder(test_output)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_patches + 1, self.embedding_dim]))

    def test_vit(self):
        test_output = self.test_vit(self.test_input)
        self.assertEqual(test_output.shape, torch.Size([self.batch_size, self.num_classes]))
