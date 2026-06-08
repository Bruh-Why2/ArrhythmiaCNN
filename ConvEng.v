`timescale 1ns / 1ps

module light_ce_8mac #(
    parameter D_WIDTH = 16,        // 16-bit fixed point for weights and inputs
    parameter ACC_WIDTH = 48       // 48-bit to prevent overflow during accumulation
)(
    input  wire                 clk,
    input  wire                 rst_n,      // Active-low reset
    input  wire                 en,         // Enable signal from the FSM
    input  wire                 clear_acc,  // FSM asserts this when starting a new dot product  
    // 8 Weight Inputs (Broadcasted from the global Weight Buffer)
    input  wire signed [D_WIDTH-1:0] w0, w1, w2, w3, w4, w5, w6, w7, 
    // 8 Activation Inputs (Read directly from the local LDM)
    input  wire signed [D_WIDTH-1:0] a0, a1, a2, a3, a4, a5, a6, a7,
    // Final accumulated output to write back to the LDM
    output reg  signed [ACC_WIDTH-1:0] out_acc
);

    // 1. Combinational Multipliers
    wire signed [ACC_WIDTH-1:0] p0 = a0 * w0;
    wire signed [ACC_WIDTH-1:0] p1 = a1 * w1;
    wire signed [ACC_WIDTH-1:0] p2 = a2 * w2;
    wire signed [ACC_WIDTH-1:0] p3 = a3 * w3;
    wire signed [ACC_WIDTH-1:0] p4 = a4 * w4;
    wire signed [ACC_WIDTH-1:0] p5 = a5 * w5;
    wire signed [ACC_WIDTH-1:0] p6 = a6 * w6;
    wire signed [ACC_WIDTH-1:0] p7 = a7 * w7;

    // 2. Combinational Adder Tree (Spatial Accumulation)
    
    // Stage 1: Add pairs together
    wire signed [ACC_WIDTH-1:0] sum_s1_0 = p0 + p1;
    wire signed [ACC_WIDTH-1:0] sum_s1_1 = p2 + p3;
    wire signed [ACC_WIDTH-1:0] sum_s1_2 = p4 + p5;
    wire signed [ACC_WIDTH-1:0] sum_s1_3 = p6 + p7;

    // Stage 2: Add the results of Stage 1
    wire signed [ACC_WIDTH-1:0] sum_s2_0 = sum_s1_0 + sum_s1_1;
    wire signed [ACC_WIDTH-1:0] sum_s2_1 = sum_s1_2 + sum_s1_3;

    // Stage 3: The final spatial sum of all 8 MACs for this specific clock cycle
    wire signed [ACC_WIDTH-1:0] tree_sum = sum_s2_0 + sum_s2_1;

    // =========================================================================
    // 3. Synchronous Accumulator (Temporal Accumulation)
    // =========================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_acc <= {ACC_WIDTH{1'b0}};
        end 
        else if (en) begin
            if (clear_acc) begin
                // FSM says this is the FIRST cycle of a new dot product.
                // We overwrite the old accumulator data with the new sum.
                out_acc <= tree_sum; 
            end 
            else begin
                // FSM says we are continuing a heavy workload (like the 56-MAC branch).
                // Add the new 8-MAC sum to the running total.
                out_acc <= out_acc + tree_sum; 
            end
        end
    end

endmodule

