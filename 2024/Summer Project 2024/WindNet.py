from Wind_backend import *


class Windnet(torch.nn.Module):
    """
    GNN model setup for offshore wind prediction.
    """
    def __init__(self, carra_input_dim=5, layers = 4, embedding_size = 64, sentinel_input_dim=32, skip=False):
        super(Windnet, self).__init__()

        self.listoflayers = torch.nn.ModuleList()
        self.skipc = skip
        # 1. Three GCNConv layers operating on CARRA nodes
        self.carra_conv1 = GCNConv(carra_input_dim, 16)
        self.carra_conv2 = GCNConv(16, 32)

        # 2. One HeteroGCNConv layer passing messages from CARRA to output grid
        self.carra_to_output = GraphConv((-1, -1), 32)

        # 3. Three GCNConv layers operating on output grid
        self.Vladimir = Linear(sentinel_input_dim, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)

        # previous = self.conv1.copy()
        if self.skipc == True:
            for i in range(layers-1):
                self.listoflayers.append(GCNConv(2*embedding_size, embedding_size))
                #Linear layer reducing output channels to one
                self.lin = Linear(2*embedding_size, 1)
        elif self.skipc == False:
            for i in range(layers):
                self.listoflayers.append(GCNConv(embedding_size, embedding_size))
                #Linear layer reducing output channels to one
                self.lin = Linear(embedding_size, 1)
    def forward(self, x_carra, x_output, edge_i_c2c, edge_i_c2o, edge_i_o2o, edge_attr_c2o, time_step):
        # 1. Three GCNConv layers operating on CARRA nodes
        
        x_carra = self.carra_conv1(x_carra, edge_i_c2c)
        x_carra = F.relu(x_carra)

        x_carra = self.carra_conv2(x_carra, edge_i_c2c)
        x_carra = F.relu(x_carra)

        # 2. One HeteroGCNConv layer passing messages from CARRA to output grid
        x_output = self.carra_to_output(
            (x_carra, x_output),
            edge_i_c2o,
            edge_weight=edge_attr_c2o
        )

        x_output = F.relu(x_output)
        if torch.isnan(x_output).any():
            print("NaNs detected after carra_to_output")

        # 3. Three GCNConv layers operating on output grid
        x_prev = self.Vladimir(x_output)
        x_prev = F.relu(x_prev)
        x_curr = self.conv1(x_prev, edge_i_o2o)
        x_curr = F.relu(x_prev)
        if self.skipc == True:
            for layer in self.listoflayers:
                x_cat = torch.cat((x_prev, x_curr), dim=1)
                x_prev = x_curr
                x_curr = layer(x_cat, edge_i_o2o)
                x_curr = F.relu(x_curr) #Du e fucked om du endre activationfunction
                x_last = torch.cat((x_prev, x_curr), dim=1)
                x_last = F.relu(x_last)

        elif self.skipc == False:
            for layer in self.listoflayers:
                x_curr = layer(x_curr, edge_i_o2o)
                x_curr = F.relu(x_curr)
                x_last = x_curr



        # 4. Linear layer reducing output channels to one
        x = self.lin(x_last)

        return x, time_step