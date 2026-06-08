`timescale 1ns / 1ps

module spatial_max_pool_8way #(
    parameter D_WIDTH = 16,
    parameter LANES   = 8
)(
    input  wire                 clk,
    input  wire                 rst_n,
    
    // Control Signals from the FSM
    input  wire                 pool_en,    // High when executing a pooling layer
    input  wire                 pool_clear, // High on the 1st cycle of a new pool window
    input  wire [2:0]           lane_sel,   // Selects which channel to write back
    
    // The massive 128-bit read bus from the LDM
    input  wire [127:0]         ldm_read_data,
    
    // The single 16-bit winning value to write back
    output wire [D_WIDTH-1:0]   pool_out
);

    // 8 parallel 16-bit registers to hold the running maximums
    reg signed [D_WIDTH-1:0] max_regs [0:LANES-1];
    wire signed [D_WIDTH-1:0] in_lanes [0:LANES-1];

    // Unpack the 128-bit bus into 8 distinct wires
    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : UNPACK
            assign in_lanes[i] = ldm_read_data[(i*D_WIDTH) +: D_WIDTH];
        end
    endgenerate

    // The Parallel Comparator Array
    integer j;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Initialize to the most negative possible 16-bit number
            for (j = 0; j < LANES; j = j + 1) max_regs[j] <= 16'h8000; 
        end 
        else if (pool_en) begin
            if (pool_clear) begin
                // First cycle of the window: blindly accept the data
                for (j = 0; j < LANES; j = j + 1) max_regs[j] <= in_lanes[j];
            end 
            else begin
                // Cycles 2 and 3: Compare and update if the new data is larger
                for (j = 0; j < LANES; j = j + 1) begin
                    if (in_lanes[j] > max_regs[j]) begin
                        max_regs[j] <= in_lanes[j];
                    end
                end
            end
        end
    end

    // Mux to select which channel is written back to the 16-bit LDM port
    assign pool_out = max_regs[lane_sel];

endmodule