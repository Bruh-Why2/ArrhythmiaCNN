`timescale 1ns / 1ps

module global_average_pool_tree #(
    parameter NUM_PES = 40,
    parameter D_WIDTH = 16,
    parameter LANES   = 8
)(
    // A massive flattened bus containing all 40 PE outputs
    input  wire [(NUM_PES*D_WIDTH*LANES)-1:0] all_pe_data,
    
    // The single 128-bit averaged output
    output wire [(D_WIDTH*LANES)-1:0]         gap_out
);

    genvar lane;
    generate
        for (lane = 0; lane < LANES; lane = lane + 1) begin : GAP_LANES
            
            // 1. Extract the 40 specific values for this single channel
            wire signed [D_WIDTH-1:0] values [0:NUM_PES-1];
            genvar pe;
            for (pe = 0; pe < NUM_PES; pe = pe + 1) begin : EXTRACT
                assign values[pe] = all_pe_data[(pe*LANES + lane)*D_WIDTH +: D_WIDTH];
            end

            // 2. Combinational Adder Tree to sum all 40 PEs
            reg signed [31:0] sum;
            integer i;
            always @(*) begin
                sum = 0;
                for (i = 0; i < NUM_PES; i = i + 1) begin
                    sum = sum + values[i];
                end
            end

            // 3. Hardware Division: Avg = (Sum * 819) >> 15
            wire signed [31:0] avg = (sum * 32'sd819) >>> 15;
            
            // 4. Pack it back into the 128-bit output bus
            assign gap_out[lane*D_WIDTH +: D_WIDTH] = avg[15:0]; 
            
        end
    endgenerate

endmodule